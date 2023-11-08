from UDL.hisr.common.main_hisr import main
from UDL.Basis.auxiliary import set_random_seed
# from UDL.hisr.HISR.Bidi_final.option_bidi import cfg as args
# from UDL.hisr.HISR.Bidi_final.model_SR import build
# from UDL.hisr.HISR.Bidi.option_bidi import cfg as args
# from UDL.hisr.HISR.Bidi.model_SR import build
# from UDL.hisr.HISR.PSRT.option_hisr_PSRT import cfg as args
# from UDL.hisr.HISR.PSRT.model_PSRT import build
# from UDL.hisr.HISR.PSRT_kernelattentionv1.option_hisr_PSRT import cfg as args
# from UDL.hisr.HISR.PSRT_kernelattentionv1.model_PSRT import build
# from UDL.hisr.HISR.PSRT_kernelattentionv2.option_hisr_PSRT import cfg as args
# from UDL.hisr.HISR.PSRT_kernelattentionv2.model_PSRT import build
# from UDL.hisr.HISR.PSRT_kernelattentionv3.option_hisr_PSRT import cfg as args
# from UDL.hisr.HISR.PSRT_kernelattentionv3.model_PSRT import build
# from UDL.hisr.HISR.PSRT_kernelattentionv4.option_hisr_PSRT import cfg as args
# from UDL.hisr.HISR.PSRT_kernelattentionv4.model_PSRT import build
# from UDL.hisr.HISR.PSRT_kernelattentionv5.option_hisr_PSRT import cfg as args
# from UDL.hisr.HISR.PSRT_kernelattentionv5.model_PSRT import build
# from UDL.hisr.HISR.PSRT_KAv1.option_hisr_PSRT import cfg as args
# from UDL.hisr.HISR.PSRT_KAv1.model_PSRT import build
# from UDL.hisr.HISR.PSRT_noshuffle.option_hisr_PSRT import cfg as args
# from UDL.hisr.HISR.PSRT_noshuffle.model_PSRT import build
# from UDL.hisr.HISR.PSRT_KAv1_noshuffle.option_hisr_PSRT import cfg as args
# from UDL.hisr.HISR.PSRT_KAv1_noshuffle.model_PSRT import build
# from UDL.hisr.HISR.PSRT_KAv2_noshuffle.option_hisr_PSRT import cfg as args
# from UDL.hisr.HISR.PSRT_KAv2_noshuffle.model_PSRT import build
# from UDL.hisr.HISR.PSRT_KAv3_noshuffle.option_hisr_PSRT import cfg as args
# from UDL.hisr.HISR.PSRT_KAv3_noshuffle.model_PSRT import build
# from UDL.hisr.HISR.PSRT_KAv4_noshuffle.option_hisr_PSRT import cfg as args
# from UDL.hisr.HISR.PSRT_KAv4_noshuffle.model_PSRT import build
# from UDL.hisr.HISR.PSRT_KAv5_noshuffle.option_hisr_PSRT import cfg as args
# from UDL.hisr.HISR.PSRT_KAv5_noshuffle.model_PSRT import build
# from UDL.hisr.HISR.PSRT_KAv6_noshuffle.option_hisr_PSRT import cfg as args
# from UDL.hisr.HISR.PSRT_KAv6_noshuffle.model_PSRT import build
# from UDL.hisr.HISR.PSRT_KAv7_noshuffle.option_hisr_PSRT import cfg as args
# from UDL.hisr.HISR.PSRT_KAv7_noshuffle.model_PSRT import build
# from UDL.hisr.HISR.PSRT_KAv8_noshuffle.option_hisr_PSRT import cfg as args
# from UDL.hisr.HISR.PSRT_KAv8_noshuffle.model_PSRT import build
# from UDL.hisr.HISR.PSRT_KAv9_noshuffle.option_hisr_PSRT import cfg as args
# from UDL.hisr.HISR.PSRT_KAv9_noshuffle.model_PSRT import build
# from UDL.hisr.HISR.PSRT_KAv10_noshuffle.option_hisr_PSRT import cfg as args
# from UDL.hisr.HISR.PSRT_KAv10_noshuffle.model_PSRT import build
# from UDL.hisr.HISR.PSRT_KAv11_noshuffle.option_hisr_PSRT import cfg as args
# from UDL.hisr.HISR.PSRT_KAv11_noshuffle.model_PSRT import build
# from UDL.hisr.HISR.PSRT_KAv12_noshuffle.option_hisr_PSRT import cfg as args
# from UDL.hisr.HISR.PSRT_KAv12_noshuffle.model_PSRT import build
# from UDL.hisr.HISR.PSRT_KAv13_noshuffle.option_hisr_PSRT import cfg as args
# from UDL.hisr.HISR.PSRT_KAv13_noshuffle.model_PSRT import build
# from UDL.hisr.HISR.PSRT_KAv14_noshuffle.option_hisr_PSRT import cfg as args
# from UDL.hisr.HISR.PSRT_KAv14_noshuffle.model_PSRT import build
from UDL.hisr.HISR.PSRT_KAv15_noshuffle.option_hisr_PSRT import cfg as args
from UDL.hisr.HISR.PSRT_KAv15_noshuffle.model_PSRT import build
# from UDL.hisr.HISR.PSRT_KAv16_noshuffle.option_hisr_PSRT import cfg as args
# from UDL.hisr.HISR.PSRT_KAv16_noshuffle.model_PSRT import build
# from UDL.hisr.HISR.PSRT_KAv16_noshuffle.option_hisr_PSRT import cfg as args
# from UDL.hisr.HISR.PSRT_KAv16_noshuffle.model_PSRT import build
from UDL.hisr.HISR.PSRT_KAv17_noshuffle.option_hisr_PSRT import cfg as args
from UDL.hisr.HISR.PSRT_KAv17_noshuffle.model_PSRT import build



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