from UDL.hisr.common.main_hisr import main
from UDL.Basis.auxiliary import set_random_seed
# from UDL.hisr.HISR.Bidi.option_bidi import cfg as args
# from UDL.hisr.HISR.Bidi.model_SR import build
# from UDL.hisr.HISR.Bidi_kernelattentionv1.option_bidi import cfg as args
# from UDL.hisr.HISR.Bidi_kernelattentionv1.model_SR import build
# from UDL.hisr.HISR.Bidi_kernelattentionv2.option_bidi import cfg as args
# from UDL.hisr.HISR.Bidi_kernelattentionv2.model_SR import build
from UDL.hisr.HISR.Bidi_kernelattentionv3.option_bidi import cfg as args
from UDL.hisr.HISR.Bidi_kernelattentionv3.model_SR import build
# from UDL.hisr.HISR.Bidi_kernelattentionv4.option_bidi import cfg as args
# from UDL.hisr.HISR.Bidi_kernelattentionv4.model_SR import build
# from UDL.hisr.HISR.Bidi_kernelattentionv5.option_bidi import cfg as args
# from UDL.hisr.HISR.Bidi_kernelattentionv5.model_SR import build
# from UDL.hisr.HISR.Bidi_kernelattentionv6.option_bidi import cfg as args
# from UDL.hisr.HISR.Bidi_kernelattentionv6.model_SR import build
# from UDL.hisr.HISR.Bidi_kernelattentionv6.option_bidi import cfg as args
# from UDL.hisr.HISR.Bidi_kernelattentionv6.model_SR import build


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