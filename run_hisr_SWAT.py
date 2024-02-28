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
# from UDL.hisr.HISR.Swin_baselinev6.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_baselinev6.model_Swin import build
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
# from UDL.hisr.HISR.Swinv8.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swinv8.model_Swin import build

# from UDL.hisr.HISR.Swin_poolv1.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv1.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv2.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv2.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv3.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv3.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv4.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv4.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv5.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv5.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv6.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv6.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv7.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv7.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv9.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv9.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv10.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv10.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv10_shortcut.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv10_shortcut.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv11.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv11.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv12.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv12.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv13.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv13.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv14.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv14.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv15.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv15.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv16.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv16.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv17.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv17.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv18.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv18.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv19.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv19.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv21.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv21.model_Swin import build

# from UDL.hisr.HISR.Swin_poolv22_normalconv.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv22_normalconv.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv22_groupconv.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv22_groupconv.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv22_groupconvfusion.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv22_groupconvfusion.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv22_groupconvfusion_beforeattn.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv22_groupconvfusion_beforeattn.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv22_groupconvfusion_shortcut.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv22_groupconvfusion_shortcut.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv22_groupconvfusion_shortcutnorm.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv22_groupconvfusion_shortcutnorm.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv23.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv23.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv24.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv24.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv24_nopoolgk.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv24_nopoolgk.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv24_nopoolgk_shortcut.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv24_nopoolgk_shortcut.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv25.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv25.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv26.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv26.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv27.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv27.model_Swin import build
from UDL.hisr.HISR.Swin_poolv28.option_hisr_Swin import cfg as args
from UDL.hisr.HISR.Swin_poolv28.model_Swin import build

# from UDL.hisr.HISR.Swin_pool_baselinev3.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_pool_baselinev3.model_Swin import build
# from UDL.hisr.HISR.Swin_poolv20.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_poolv20.model_Swin import build

# from UDL.hisr.HISR.Swin_pool_baselinev2.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_pool_baselinev2.model_Swin import build
# from UDL.hisr.HISR.Swin_qkvv1.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_qkvv1.model_Swin import build
# from UDL.hisr.HISR.Swin_qkvv2.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_qkvv2.model_Swin import build
# from UDL.hisr.HISR.Swin_qkvv3.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_qkvv3.model_Swin import build
# from UDL.hisr.HISR.Swin_qkvv4.option_hisr_Swin import cfg as args
# from UDL.hisr.HISR.Swin_qkvv4.model_Swin import build



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