from UDL.hisr.common.main_hisr import main
from UDL.Basis.auxiliary import set_random_seed
# from UDL.hisr.HISR.SWAT_baseline.option_hisr_SWAT import cfg as args
# from UDL.hisr.HISR.SWAT_baseline.model_SWAT import build
# from UDL.hisr.HISR.SWAT_baselinev2.option_hisr_SWAT import cfg as args
# from UDL.hisr.HISR.SWAT_baselinev2.model_SWAT import build
# from UDL.hisr.HISR.SWAT_baseline_noshift.option_hisr_SWAT import cfg as args
# from UDL.hisr.HISR.SWAT_baseline_noshift.model_SWAT import build
# from UDL.hisr.HISR.SWAT_baseline_noshiftv4.option_hisr_SWAT import cfg as args
# from UDL.hisr.HISR.SWAT_baseline_noshiftv4.model_SWAT import build
# from UDL.hisr.HISR.SWATv1.option_hisr_SWAT import cfg as args
# from UDL.hisr.HISR.SWATv1.model_SWAT import build
# from UDL.hisr.HISR.SWATv2.option_hisr_SWAT import cfg as args
# from UDL.hisr.HISR.SWATv2.model_SWAT import build
# from UDL.hisr.HISR.SWATv3.option_hisr_SWAT import cfg as args
# from UDL.hisr.HISR.SWATv3.model_SWAT import build
# from UDL.hisr.HISR.SWATv4.option_hisr_SWAT import cfg as args
# from UDL.hisr.HISR.SWATv4.model_SWAT import build

# from UDL.hisr.HISR.Swin_baseline.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_baseline.model_Swin import build
# from UDL.hisr.HISR.Swinv1.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swinv1.model_Swin import build
# from UDL.hisr.HISR.Swin_baselinev2.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_baselinev2.model_Swin import build
# from UDL.hisr.HISR.Swin_baselinev3.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_baselinev3.model_Swin import build
from UDL.hisr.HISR.Swin_baselinev6.option_hisr_Swin import cfg as args
from UDL.hisr.HISR.Swin_baselinev6.model_Swin import build
# from UDL.hisr.HISR.Swinv3.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swinv3.model_Swin import build
# from UDL.hisr.HISR.Swinv4.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swinv4.model_Swin import build

# from UDL.hisr.HISR.Swinv6.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swinv6.model_Swin import build
# from UDL.hisr.HISR.Swinv7.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swinv7.model_Swin import build
# from UDL.hisr.HISR.Swinv8.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swinv8.model_Swin import build



import os
if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # cfg = Config.fromfile("../pansharpening/DCFNet/option_DCFNet.py")
    set_random_seed(args.seed)
    # print(cfg.builder)
    # print(model_DFTL_v1)
    # log_string(args)
    args.builder = build
    main(args)
#
# if __name__ == '__main__':
#     import glob
#     import shutil
#     fdir = "./my_model_results/Rain200L/"
#     flist = glob.glob(fdir + "*.png")
#     for file in flist:
#         print(file)
#         # shutil.move(file, file.split('.')[0][:-3]+'.png')