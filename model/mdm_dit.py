import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import clip
from model.rotation2xyz import Rotation2xyz
from typing import Optional, Tuple


def modulate(x, shift, scale):
    # support shift and without shift
    return x * (1 + scale) + (shift if shift is not None else 0)


class DiTBlockConcat(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.linear0 = nn.Linear(d_model * 2, d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm0 = nn.LayerNorm(d_model * 2)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(d_model, 6 * d_model, bias=True))
        # set adaLN's parameters to zero (adaLN-Zero)
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self,
                src,
                c,
                skip,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None):
        """
        Args: 
            src: [t, bs, d_model]
            c: [1, bs, d_model]
        """
        assert len(c.shape) == 3
        # [1, bs, d_model * 2]
        scale_in0, scale_in1, shift_msa, scale_msa, gate_msa, gate_mlp = self.adaLN_modulation(
            c).chunk(6, dim=-1)

        # concat skip connection and input
        # [t, bs, d_model * 2]
        src = torch.cat([src, skip], dim=-1)
        src = modulate(self.norm0(src), None,
                       torch.cat([scale_in0, scale_in1], dim=-1))
        # [t, bs, d_model]
        src = self.linear0(src)

        x = gate_msa * self.attn(src,
                                 src,
                                 src,
                                 attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(x)
        src = modulate(self.norm1(src), shift_msa, scale_msa)

        x = gate_mlp * self.linear2(
            self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(x)
        # NOTE: modulation is not applied to the output because it is already applied to the input
        # this model is a kind of pre-norm
        return src


class DiTBlockConcatV2(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 scale_only=False):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # combines skip connection and input
        self.linear1 = nn.Linear(d_model * 2, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.scale_only = scale_only
        adaLN_width = 4 * d_model if self.scale_only else 6 * d_model
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(d_model, adaLN_width, bias=True))
        # set adaLN's parameters to zero (adaLN-Zero)
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self,
                src,
                c,
                skip,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None):
        """
        Args: 
            src: [t, bs, d_model]
            c: [1, bs, d_model]
        """
        assert len(c.shape) == 3
        if self.scale_only:
            # [1, bs, d_model * 2]
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(
                c).chunk(4, dim=-1)
            shift_msa = shift_mlp = None
        else:
            # [1, bs, d_model * 2]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
                c).chunk(6, dim=-1)

        x = gate_msa * self.attn(src,
                                 src,
                                 src,
                                 attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(x)
        src = modulate(self.norm1(src), shift_msa, scale_msa)

        # combines skip connection and input
        x = gate_mlp * self.linear2(
            self.dropout(
                self.activation(self.linear1(torch.cat([src, skip], dim=-1)))))
        src = src + self.dropout2(x)
        src = modulate(self.norm2(src), shift_mlp, scale_mlp)
        return src


class DiTBlockPostNorm(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    following PyTorch's TransformerEncoderLayer.
    """
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(d_model, 6 * d_model, bias=True))
        # set adaLN's parameters to zero (adaLN-Zero)
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self,
                src,
                c,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                **kwargs):
        """
        Args: 
            src: [t, bs, d_model]
            c: [1, bs, d_model]
        """
        assert len(c.shape) == 3
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c).chunk(6, dim=-1)

        x = gate_msa * self.attn(src,
                                 src,
                                 src,
                                 attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(x)
        src = modulate(self.norm1(src), shift_msa, scale_msa)

        x = gate_mlp * self.linear2(
            self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(x)
        src = modulate(self.norm2(src), shift_mlp, scale_mlp)
        return src


class DiTBlockPreNorm(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    following PyTorch's TransformerEncoderLayer.
    """
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(d_model, 6 * d_model, bias=True))
        # set adaLN's parameters to zero (adaLN-Zero)
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self,
                src,
                c,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                **kwargs):
        """
        Args: 
            src: [t, bs, d_model]
            c: [1, bs, d_model]
        """
        assert len(c.shape) == 3
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c).chunk(6, dim=-1)

        x = modulate(self.norm1(src), shift_msa, scale_msa)
        x = gate_msa * self.attn(
            x, x, x, attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(x)

        x = modulate(self.norm2(src), shift_mlp, scale_mlp)
        x = gate_mlp * self.linear2(
            self.dropout(self.activation(self.linear1(x))))
        src = src + self.dropout2(x)
        return src


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(
        "activation should be relu/gelu, not {}".format(activation))


class DiTEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(DiTEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                src: Tensor,
                c: Tensor,
                skip: Tensor = None,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output,
                         c=c,
                         skip=skip,
                         src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MDM_DiT(nn.Module):
    """
    DiT-style MDM that improves time condition.
    """
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
                 two_head=False,
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

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.two_head = two_head
        if two_head:
            print('Using two head MDM')

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.input_process = InputProcess(self.data_rep,
                                          self.input_feats + self.gru_emb_dim,
                                          self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim,
                                                       self.dropout)
        self.emb_trans_dec = emb_trans_dec

        scale_only = 'scale' in self.arch
        if self.arch.startswith('dit_prenorm'):
            print("DiT prenorm")
            seqTransEncoderLayer = DiTBlockPreNorm(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation)
            add_norm_before_pred = True
            use_skip_connection = False
        elif self.arch.startswith('dit_postnorm'):
            print("DiT postnorm")
            seqTransEncoderLayer = DiTBlockPostNorm(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation)
            add_norm_before_pred = False
            use_skip_connection = False
        elif self.arch.startswith('dit_concatv2'):
            # needs to be before dit_concat due to the common prefix
            print(f"DiT concat v2 (scale only: {scale_only})")
            # supports 'dit_concatv2_scale'
            seqTransEncoderLayer = DiTBlockConcatV2(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation,
                scale_only=scale_only)
            add_norm_before_pred = False
            use_skip_connection = True
        elif self.arch.startswith('dit_concatv3'):
            # the same as V2 but doesn't use the skip connection at the output module
            # needs to be before dit_concat due to the common prefix
            print(f"DiT concat v3 (scale only: {scale_only})")
            # supports 'dit_concatv2_scale'
            seqTransEncoderLayer = DiTBlockConcatV2(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation,
                scale_only=scale_only)
            add_norm_before_pred = False
            use_skip_connection = False
        elif self.arch.startswith('dit_concat'):
            print("DiT concat")
            seqTransEncoderLayer = DiTBlockConcat(d_model=self.latent_dim,
                                                  nhead=self.num_heads,
                                                  dim_feedforward=self.ff_size,
                                                  dropout=self.dropout,
                                                  activation=self.activation)
            add_norm_before_pred = True
            use_skip_connection = True
        else:
            raise ValueError(
                'Please choose correct architecture [trans_enc, trans_dec, gru]'
            )

        # for predicting eps
        self.output_process = OutputProcess(self.data_rep,
                                            self.input_feats,
                                            self.latent_dim,
                                            self.njoints,
                                            self.nfeats,
                                            norm=add_norm_before_pred,
                                            zero=True,
                                            skip=use_skip_connection,
                                            scale_only=scale_only)
        if self.two_head:
            # predicting x0
            self.output_process2 = OutputProcess(self.data_rep,
                                                 self.input_feats,
                                                 self.latent_dim,
                                                 self.njoints,
                                                 self.nfeats,
                                                 norm=add_norm_before_pred,
                                                 zero=True,
                                                 skip=use_skip_connection,
                                                 scale_only=scale_only)

        self.seqTransEncoder = DiTEncoder(seqTransEncoderLayer,
                                          num_layers=self.num_layers)

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

    def forward(self, x, timesteps, y=None):
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

        x = self.input_process(x)

        # adding the timestep embed
        skip = x = self.sequence_pos_encoder(x)  # [seqlen, bs, d]
        x = self.seqTransEncoder(
            x, c=emb,
            skip=skip)  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        output = self.output_process(
            x, c=emb, skip=skip)  # [bs, njoints, nfeats, nframes]
        if self.two_head:
            # will also predict x0
            output2 = self.output_process2(x, c=emb, skip=skip)
            return output, output2
        else:
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


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_channels,
                 norm=True,
                 zero=True,
                 scale_only=False):
        super().__init__()
        self.norm_final = nn.LayerNorm(in_channels,
                                       elementwise_affine=False,
                                       eps=1e-6) if norm else nn.Identity()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.scale_only = scale_only
        adaLN_width = in_channels if self.scale_only else 2 * in_channels
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(cond_channels, adaLN_width, bias=True))
        if zero:
            # zero init the output layer
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
            nn.init.zeros_(self.adaLN_modulation[1].weight)
            nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        """
        Args:
            x: [seqlen, bs, d]
            c: [1, bs, d]
        """
        assert len(c.shape) == 3
        if self.scale_only:
            scale = self.adaLN_modulation(c)
            shift = None
        else:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class OutputProcess(nn.Module):
    def __init__(self,
                 data_rep,
                 input_feats,
                 latent_dim,
                 njoints,
                 nfeats,
                 norm,
                 zero,
                 skip=False,
                 scale_only=False):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.skip = skip

        # with skip connection the output dimension is doubled
        self.effective_dim = self.latent_dim * 2 if self.skip else self.latent_dim
        self.poseFinal = FinalLayer(self.effective_dim,
                                    self.input_feats,
                                    self.latent_dim,
                                    norm,
                                    zero=zero,
                                    scale_only=scale_only)
        if self.data_rep == 'rot_vel':
            self.velFinal = FinalLayer(self.effective_dim,
                                       self.input_feats,
                                       self.latent_dim,
                                       norm,
                                       zero=zero,
                                       scale_only=scale_only)

    def forward(self, output, c, skip=None):
        if self.skip:
            # concat output with skip connection
            output = torch.cat([output, skip], dim=-1)

        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output, c=c)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose, c=c)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel, c=c)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
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