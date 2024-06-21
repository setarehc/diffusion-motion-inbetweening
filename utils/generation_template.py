from utils.parser_util import FullModelArgs


def get_template(args: FullModelArgs, template_name="no"):
    # [no, trajectory, kps, sdf]
    if template_name == "mdm_legacy":
        updated_args = mdm_template(args)
    elif template_name == "no":
        updated_args = args
    elif template_name == "trajectory":
        updated_args = trajectory_template(args)
    elif template_name == "kps":
        updated_args = kps_template(args)
    elif template_name == "sdf":
        updated_args = sdf_template(args)
    elif template_name == "testing":
        updated_args = testing_template(args)
    else:
        raise NotImplementedError()
    return updated_args


def mdm_template(args: FullModelArgs):
    # NOTE: backward compatible. Otherwise, get_template() is only allowed to change generate args.
    MODEL_NAME = 'motion,trans_enc,x0,rel,normal'.split(',')
    # args.gen_avg_model = False
    args.motion_length = 6.0
    args.abs_3d = False
    args.gen_two_stages = False
    args.do_inpaint = True
    # This "mdm_legacy" mode is only used when we do trajectory imputing with mdm
    args.guidance_mode = "mdm_legacy"

    return args


def trajectory_template(args: FullModelArgs):
    args.do_inpaint = True
    # Data flags  
    # NOTE: this should already be in json for new model
    # May need to update json for previous model
    # args.use_random_proj = True
    # args.random_proj_scale = 10.0
    args.guidance_mode = "trajectory"  # ["no", "trajectory", "kps", "sdf"]
    args.gen_two_stages = False
    return args


def kps_template(args: FullModelArgs):
    args.do_inpaint = True
    args.guidance_mode = "kps"  # ["no", "trajectory", "kps", "sdf"]
    args.gen_two_stages = True
    # NOTE: set imputation p2p mode here
    # args.p2p_impute = False
    args.p2p_impute = True

    return args


def sdf_template(args: FullModelArgs):
    args.do_inpaint = True
    args.guidance_mode = "sdf"  # ["no", "trajectory", "kps", "sdf"]
    args.gen_two_stages = True
    args.p2p_impute = False
    return args


def testing_template(args: FullModelArgs):
    args.do_inpaint = False # True
    args.guidance_mode = "no"  # ["no", "trajectory", "kps", "sdf"]
    # args.classifier_scale = 1.0
    args.gen_two_stages = False
    args.p2p_impute = False
    args.use_ddim = False # True
    # args.motion_length = 4.5
    args.interpolate_cond = False # True
    return args