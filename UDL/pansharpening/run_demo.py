from UDL.Basis.config import Config
from UDL.pansharpening.common.main_pansharpening import main
from UDL.Basis.auxiliary import set_random_seed
from UDL.pansharpening.models.Dilation.option_dilation import cfg as args
from UDL.pansharpening.models.Dilation.dilation_main import build

if __name__ == '__main__':
    # cfg = Config.fromfile("../pansharpening/DCFNet/option_DCFNet.py")
    set_random_seed(args.seed)
    # print(cfg.builder)
    args.builder = build
    main(args)