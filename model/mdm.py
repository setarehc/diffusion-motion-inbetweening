import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.rotation2xyz import Rotation2xyz
from einops.layers.torch import Rearrange

class MDM(nn.Module):
    def __init__(self,
                 modeltype,
                 njoints,
                 nfeats,
                 num_actions,
                 translation,
                 pose_rep,
                 glob,
                 glob_rot,
                 latent_dim=256,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=4,
                 dropout=0.1,
                 ablation=None,
                 activation="gelu",
                 legacy=False,
                 data_rep='rot6d',
                 dataset='amass',
                 clip_dim=512,
                 arch='trans_enc',
                 emb_trans_dec=False,
                 clip_version=None,
                 tf_out_mult=None,
                 train_keypoint_mask='none',
                 **kargs):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        print('MDM latent dim: ', latent_dim)
        self.latent_dim = latent_dim
        print('MDM ff size: ', ff_size)
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats
        self.train_keypoint_mask = train_keypoint_mask
        
        self.better_cond = True
        self.latent_dim_input = self.latent_dim
        if 'better_cond' in self.train_keypoint_mask:
            self.added_input_dim = 0
            # If use better condition, we will enlarge the transformer input dim by 64
            # # This will require some change to the code
            # Change latent_dim for encoder layer
            self.cond_latent_dim = 64
            if 'keypoints' in self.train_keypoint_mask:
                self.cond_dim = 2
            # Create new projection layer before PE and transformer encoder
            # The output of this layer will be masked
            self.cond_process = CondProcess(self.cond_dim, self.cond_latent_dim)
            print('add cond latent to MDM latent. New dim is: %d + %d' % (self.latent_dim, self.cond_latent_dim))
            self.latent_dim = self.latent_dim + self.cond_latent_dim
            
        elif self.train_keypoint_mask == 'keypoints':
            self.added_input_dim = 3
        elif self.train_keypoint_mask == 'keyposes':
            self.added_input_dim = 68
        else:
            self.added_input_dim = 0

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.input_process = InputProcess(self.data_rep,
                                          self.input_feats + self.gru_emb_dim + self.added_input_dim,
                                          self.latent_dim_input)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim,
                                                       self.dropout)
        self.emb_trans_dec = emb_trans_dec

        if self.arch.startswith('trans_enc'):
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation)
            self.seqTransEncoder = nn.TransformerEncoder(
                seqTransEncoderLayer, num_layers=self.num_layers)
        elif self.arch.startswith('trans_dec'):
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(
                seqTransDecoderLayer, num_layers=self.num_layers)
        elif self.arch.startswith('gru'):
            print("GRU init")
            self.gru = nn.GRU(self.latent_dim,
                              self.latent_dim,
                              num_layers=self.num_layers,
                              batch_first=True)
        else:
            raise ValueError(
                'Please choose correct architecture [trans_enc, trans_dec, gru]'
            )

        self.embed_timestep = TimestepEmbedder(self.latent_dim,
                                               self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print('EMBED TEXT')
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)
            if 'action' in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions,
                                                self.latent_dim)
                print('EMBED ACTION')

        if self.arch.endswith('_large'):
            print('Using large output process with tf_out_mults: ',
                  tf_out_mult)
            self.output_process = OutputProcessLarge(self.data_rep,
                                                     self.input_feats,
                                                     self.latent_dim,
                                                     self.njoints, self.nfeats,
                                                     tf_out_mult)
        else:
            self.output_process = OutputProcess(self.data_rep,
                                                self.input_feats,
                                                self.latent_dim, self.njoints,
                                                self.nfeats)

        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def parameters_wo_clip(self):
        return [
            p for name, p in self.named_parameters()
            if not name.startswith('clip_model.')
        ]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(
            clip_version, device='cpu',
            jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model
        )  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(
                    bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
    
    def mask_kps_cond(self, xseq, cond_mask):
        # Beware that the first position is now text embedding.
        # Masking needed to be shifted by one position
        # xseq - [seq_len + 1, bs, latent_dim_w_cond]
        bs, njoints, nfeats, nframes = cond_mask.shape
        cond_mask = cond_mask.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)
        # cond_mask - [seq_len, bs, 1]
        # Mask the conditional dimensions
        xseq[1:, :, -self.cond_latent_dim:] = xseq[1:, :, -self.cond_latent_dim:] * (cond_mask)
        return xseq

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in [
            'humanml', 'kit'
        ] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2  # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(
                raw_text, context_length=context_length, truncate=True
            ).to(
                device
            )  # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros(
                [texts.shape[0], default_context_length - context_length],
                dtype=texts.dtype,
                device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(
                device
            )  # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    def forward(self, x, timesteps, y=None, cond_val=None, cond_mask=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(
                self.mask_cond(enc_text, force_mask=force_mask))
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)

        if self.arch == 'gru':
            x_reshaped = x.reshape(bs, njoints * nfeats, 1, nframes)
            emb_gru = emb.repeat(nframes, 1, 1)  #[#frames, bs, d]
            emb_gru = emb_gru.permute(1, 2, 0)  #[bs, d, #frames]
            emb_gru = emb_gru.reshape(bs, self.latent_dim, 1,
                                      nframes)  #[bs, d, 1, #frames]
            x = torch.cat((x_reshaped, emb_gru),
                          axis=1)  #[bs, d+joints*feat, 1, #frames]

        # source input for skip connection
        bs, njoints, nfeats, nframes = x.shape
        src = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)

        # the usual input to the model
        # will be linear projected
        x = self.input_process(x)

        if self.arch.startswith('trans_enc'):
            if 'better_cond' in self.train_keypoint_mask:
                # adding the condition to x
                cond_val = self.cond_process(cond_val)
                x = torch.cat((x, cond_val), axis=2)  # [seqlen, bs, input_d + cond_d]
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            if 'better_cond' in self.train_keypoint_mask:
                # mask non-conditioned parts according to masked frames
                xseq = self.mask_kps_cond(xseq, cond_mask)
            output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        elif self.arch.startswith('trans_dec'):
            if self.emb_trans_dec:
                xseq = torch.cat((emb, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            if self.emb_trans_dec:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)[
                    1:]  # [seqlen, bs, d] # FIXME - maybe add a causal mask
            else:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)
        elif self.arch.startswith('gru'):
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
            output, _ = self.gru(xseq)
        else:
            raise NotImplementedError()

        output = self.output_process(
            output, skip=src)  # [bs, njoints, nfeats, nframes]
        return output

    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(
            self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class CondProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.condEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)
        x = self.condEmbedding(x)  # [seqlen, bs, d]
        return x


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output, **kwargs):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


def cal_multiple(n, multiple):
    """
    calculate the output channels while keeping the output channels a multiple of the given number
    """
    if n % multiple == 0:
        return n
    a = n / multiple
    return int((1 - (a - math.floor(a))) * multiple + n)


def interleave(a, b, groups):
    """
    Interleave two tensors into a number of groups along the second dimension.
    Args:
        a: (bs, n, dim)
        b: (bs, m, dim)
        where n and m are divisible by groups
    """
    assert a.shape[0] == b.shape[0]
    assert a.shape[2] == b.shape[2]
    bs, _, dim = a.shape
    # the trick is to reshape => concat => reshape.
    c = torch.cat([a.reshape(bs, groups, -1),
                   b.reshape(bs, groups, -1)],
                  dim=-1)
    c = c.reshape(bs, -1, dim)
    return c


class OutputProcessLarge(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats,
                 nmults):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.nmults = nmults

        self.large_out_type = 4
        # NOTE: not sure about the different types of output processing
        if self.large_out_type == 1:
            final_in = cal_multiple(latent_dim + input_feats, nmults)
            self.final_conv = nn.Sequential(
                Rearrange('s b d -> b d s'),
                # combine the skip connection with the upstream feature
                # randomly arrange the channels
                nn.Conv1d(latent_dim + input_feats, final_in, 1),
                # [batch, mult * in_dim, seqlen]
                nn.Conv1d(final_in,
                          nmults * input_feats,
                          5,
                          padding=2,
                          groups=nmults),
                nn.Mish(),
                nn.Conv1d(nmults * input_feats,
                          input_feats,
                          1,
                          groups=input_feats),
                Rearrange('b d s -> s b d'),
            )
        elif self.large_out_type == 2:
            final_in = cal_multiple(latent_dim + input_feats, nmults)
            self.final_conv = nn.Sequential(
                Rearrange('s b d -> b d s'),
                # combine the skip connection with the upstream feature
                # randomly arrange the channels
                nn.Conv1d(latent_dim + input_feats, final_in, 1),
                # [batch, mult * in_dim, seqlen]
                nn.Conv1d(final_in,
                          nmults * input_feats,
                          5,
                          padding=2,
                          groups=nmults),
                nn.Mish(),
                nn.Conv1d(nmults * input_feats,
                          input_feats,
                          5,
                          padding=2,
                          groups=input_feats),
                Rearrange('b d s -> s b d'),
            )
        elif self.large_out_type == 4:
            self.skip_conv = nn.Sequential(
                Rearrange('s b d -> b d s'),
                nn.Conv1d(input_feats,
                          nmults * input_feats,
                          5,
                          padding=2,
                          groups=input_feats),
            )
            latent_in = cal_multiple(latent_dim, input_feats)
            self.latent_conv = nn.Sequential(
                Rearrange('s b d -> b d s'),
                nn.Conv1d(latent_dim, latent_in, 1) if latent_dim != latent_in else nn.Identity(),
                nn.Conv1d(latent_in,
                          nmults * latent_in,
                          5,
                          padding=2,
                          groups=input_feats),
            )
            self.final_conv = nn.Sequential(
                nn.Conv1d(nmults * (input_feats + latent_in),
                          nmults * input_feats,
                          5,
                          padding=2,
                          groups=input_feats),
                nn.Mish(),
                nn.Conv1d(nmults * input_feats,
                          input_feats,
                          1,
                          groups=input_feats),
                Rearrange('b d s -> s b d'),
            )
        elif self.large_out_type == 5:
            # multiple stages
            raise NotImplementedError()
            final_in = cal_multiple(latent_dim + input_feats, nmults)
            self.final_conv = nn.Sequential(
                Rearrange('s b d -> b d s'),
                # combine the skip connection with the upstream feature
                # randomly arrange the channels
                nn.Conv1d(latent_dim + input_feats, final_in, 1),
                # [batch, mult * in_dim, seqlen]
                nn.Conv1d(final_in,
                          nmults * input_feats,
                          5,
                          padding=2,
                          groups=nmults),
                nn.Mish(),
                nn.Conv1d(nmults * input_feats,
                          input_feats,
                          1,
                          groups=input_feats),
                Rearrange('b d s -> s b d'),
            )
        else:
            raise NotImplementedError()

    def forward(self, output, skip):
        """
        Args:
            output: [seqlen, bs, d1]
            skip: [seqlen, bs, d2]
        """
        nframes, bs, _ = output.shape
        out_feat = skip.shape[-1]
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            if self.large_out_type == 4:
                # [bs, d1, seqlen]
                skip = self.skip_conv(skip)
                # [bs, d2, seqlen]
                output = self.latent_conv(output)
                # [bs, d1 + d2, seqlen]
                output = interleave(output, skip, groups=out_feat)
                # [seqlen, bs, d]
                output = self.final_conv(output)
            else:
                output = torch.concat([output, skip], dim=2)
                output = self.final_conv(output)
        else:
            raise NotImplementedError()
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(
            torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output