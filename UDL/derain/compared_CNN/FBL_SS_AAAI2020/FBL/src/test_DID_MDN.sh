CUDA_VISIBLE_DEVICES=1 python main.py --dir_data /data/yangwenhan/DID-MDN-training/ --data_train RainDid --data_test RainDidTest --data_range 1-12000/1-100 --scale 2 --model RFBLL --patch_size 64 --test_only --save_results --pre_train ../experiment/RFBLL_DID_MDN/model/model_best.pt --save RFBLL_DID_MDN_test
