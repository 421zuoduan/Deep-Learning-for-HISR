from UDL.Basis.config import Config
from UDL.derain.common.main_derain import main
from UDL.Basis.auxiliary import set_random_seed
from UDL.derain.compared_CNN.FuGCN.pytorch.option_FuGCN import cfg as args
from UDL.derain.compared_CNN.FuGCN.pytorch.model_fu import build_FuGCN as builder

if __name__ == '__main__':
    # cfg = Config.fromfile("../pansharpening/DCFNet/option_DCFNet.py")
    set_random_seed(args.seed)
    # print(cfg.builder)
    args.builder = builder
    main(args)