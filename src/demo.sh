#CUDA_VISIBLE_DEVICES=0 python3.7 main.py --scale 4 --test_scale 4 --save EMSRDPN_ResNet_BIx4 --model EMSRDPN --epochs 1250 --batch_size 16 --patch_size 48 --n_GPUs 1 --n_threads 16 --SRDPNconfig B --ext sep --data_test Set5 --reset --decay 250-500-750-1000-1250 --lr_patch_size --data_range 1-3450 --data_train Flickr2K

#CUDA_VISIBLE_DEVICES=1 python3.7 main.py --scale 4 --test_scale 4 --save EMSRDPN_ResNet_BIx4_latest --model EMSRDPN --epochs 1250 --batch_size 16 --patch_size 48 --n_GPUs 1 --n_threads 16 --SRDPNconfig B --ext sep --data_test Set5+Set14+B100+Urban100+Manga109 --decay 250-500-750-1000-1250 --lr_patch_size --data_range 1-3450 --data_train Flickr2K --test_only --pre_train ../experiment/EMSRDPN_ResNet_BIx4/model/model_latest.pt --save_results --dir_data /data/SR


#CUDA_VISIBLE_DEVICES=0 python3.7 main.py --scale 4 --test_scale 4 --save EMSRDPN_DenseNet_BIx4 --model EMSRDPN --epochs 1250 --batch_size 16 --patch_size 48 --n_GPUs 1 --n_threads 16 --SRDPNconfig C --ext sep --data_test Set5 --reset --decay 250-500-750-1000-1250 --lr_patch_size --data_range 1-3450 --data_train Flickr2K

#CUDA_VISIBLE_DEVICES=0 python3.7 main.py --scale 4 --test_scale 4 --save EMSRDPN_DenseNet_BIx4_latest --model EMSRDPN --epochs 1250 --batch_size 16 --patch_size 48 --n_GPUs 1 --n_threads 16 --SRDPNconfig C --ext sep --data_test Set5+Set14+B100+Urban100+Manga109 --decay 250-500-750-1000-1250 --lr_patch_size --data_range 1-3450 --data_train Flickr2K --test_only --pre_train ../experiment/EMSRDPN_DenseNet_BIx4/model/model_latest.pt --save_results --dir_data /data/SR


#CUDA_VISIBLE_DEVICES=1 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348_DIV2K --model EMSRDPN --epochs 4000 --batch_size 16 --patch_size 48 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --reset --decay 800-1600-2400-3200-4000 --lr_patch_size --data_range 1-800 --data_train DIV2K

#CUDA_VISIBLE_DEVICES=1 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348_DIV2K --model EMSRDPN --epochs 4000 --batch_size 16 --patch_size 48 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --resume -1 --decay 800-1600-2400-3200-4000 --lr_patch_size --data_range 1-800 --data_train DIV2K --load EMSRDPN_BIx2348_DIV2K

#CUDA_VISIBLE_DEVICES=1 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348_DIV2K_latest --model EMSRDPN --epochs 4000 --batch_size 16 --patch_size 48 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --decay 800-1600-2400-3200-4000 --lr_patch_size --data_range 1-800 --data_train DIV2K --pre_train ../experiment/EMSRDPN_BIx2348_DIV2K/model/model_latest.pt --test_only --data_test Set5+Set14+B100+Urban100+Manga109 --save_results --dir_data /data/SR

#CUDA_VISIBLE_DEVICES=1 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348_DIV2K_latest+ --model EMSRDPN --epochs 4000 --batch_size 16 --patch_size 48 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --decay 800-1600-2400-3200-4000 --lr_patch_size --data_range 1-800 --data_train DIV2K --pre_train ../experiment/EMSRDPN_BIx2348_DIV2K/model/model_latest.pt --test_only --data_test Set5+Set14+B100+Urban100+Manga109 --save_results --dir_data /data/SR --self_ensemble

#CUDA_VISIBLE_DEVICES=1 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348_DIV2K_best_Set5x2 --model EMSRDPN --epochs 4000 --batch_size 16 --patch_size 48 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --decay 800-1600-2400-3200-4000 --lr_patch_size --data_range 1-800 --data_train DIV2K --pre_train ../experiment/EMSRDPN_BIx2348_DIV2K/model/model_best_Set5x2.pt --test_only --data_test Set5+Set14+B100+Urban100+Manga109
#
#CUDA_VISIBLE_DEVICES=1 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348_DIV2K_best_Set5x3 --model EMSRDPN --epochs 4000 --batch_size 16 --patch_size 48 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --decay 800-1600-2400-3200-4000 --lr_patch_size --data_range 1-800 --data_train DIV2K --pre_train ../experiment/EMSRDPN_BIx2348_DIV2K/model/model_best_Set5x3.pt --test_only --data_test Set5+Set14+B100+Urban100+Manga109
#
#CUDA_VISIBLE_DEVICES=1 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348_DIV2K_best_Set5x4 --model EMSRDPN --epochs 4000 --batch_size 16 --patch_size 48 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --decay 800-1600-2400-3200-4000 --lr_patch_size --data_range 1-800 --data_train DIV2K --pre_train ../experiment/EMSRDPN_BIx2348_DIV2K/model/model_best_Set5x4.pt --test_only --data_test Set5+Set14+B100+Urban100+Manga109
#
#CUDA_VISIBLE_DEVICES=1 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348_DIV2K_best_Set5x8 --model EMSRDPN --epochs 4000 --batch_size 16 --patch_size 48 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --decay 800-1600-2400-3200-4000 --lr_patch_size --data_range 1-800 --data_train DIV2K --pre_train ../experiment/EMSRDPN_BIx2348_DIV2K/model/model_best_Set5x8.pt --test_only --data_test Set5+Set14+B100+Urban100+Manga109


CUDA_VISIBLE_DEVICES=0,1 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348 --model EMSRDPN --epochs 5000 --batch_size 16 --patch_size 48 --n_GPUs 2 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --reset --decay 1000-2000-3000-4000-5000 --lr_patch_size --data_range 1-3450 --data_train Flickr2K

#CUDA_VISIBLE_DEVICES=0,1 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348 --model EMSRDPN --epochs 5000 --batch_size 16 --patch_size 48 --n_GPUs 2 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --resume -1 --decay 1000-2000-3000-4000-5000 --lr_patch_size --data_range 1-3450 --data_train Flickr2K --load EMSRDPN_BIx2348

#CUDA_VISIBLE_DEVICES=0 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348_test --model EMSRDPN --epochs 5000 --batch_size 16 --patch_size 48 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5+Set14+B100+Urban100+Manga109 --reset --decay 1000-2000-3000-4000-5000 --lr_patch_size --data_range 1-3450 --data_train Flickr2K --pre_train ../experiment/EMSRDPN_BIx2348/model/model_latest.pt --test_only --save_results

#CUDA_VISIBLE_DEVICES=0 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348_test+ --model EMSRDPN --epochs 5000 --batch_size 16 --patch_size 48 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5+Set14+B100+Urban100+Manga109 --reset --decay 1000-2000-3000-4000-5000 --lr_patch_size --data_range 1-3450 --data_train Flickr2K --pre_train ../experiment/EMSRDPN_BIx2348/model/model_latest.pt --test_only --save_results --self_ensemble

#CUDA_VISIBLE_DEVICES=0 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348_test_multi_scale_infer --model EMSRDPN --epochs 5000 --batch_size 16 --patch_size 48 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --reset --decay 1000-2000-3000-4000-5000 --lr_patch_size --data_range 1-3450 --data_train Flickr2K --pre_train ../experiment/EMSRDPN_BIx2348/model/model_latest.pt --test_only --save_results --multi_scale_infer

#CUDA_VISIBLE_DEVICES=6,7 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save RDN_MSL_BIx2348 --model RDN_MSL --epochs 5000 --batch_size 16 --patch_size 48 --n_GPUs 2 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --reset --decay 1000-2000-3000-4000-5000 --lr_patch_size --data_range 1-3450 --data_train Flickr2K

#CUDA_VISIBLE_DEVICES=6,7 python3.7 main.py --scale 2+3+4 --test_scale 2+3+4 --save RDN_MSL_BIx234 --model RDN_MSL --epochs 3750 --batch_size 16 --patch_size 48 --n_GPUs 2 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --reset --decay 750-1500-2250-3000-3750 --lr_patch_size --data_range 1-3450 --data_train Flickr2K

#CUDA_VISIBLE_DEVICES=6,7 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save RDN_MSL_BIx2348_DIV2K --model RDN_MSL --epochs 4000 --batch_size 16 --patch_size 48 --n_GPUs 2 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --reset --decay 800-1600-2400-3200-4000 --lr_patch_size --data_range 1-800 --data_train DIV2K

#CUDA_VISIBLE_DEVICES=6,7 python3.7 main.py --scale 2+3+4 --test_scale 2+3+4 --save RDN_MSL_BIx234_DIV2K --model RDN_MSL --epochs 3000 --batch_size 16 --patch_size 48 --n_GPUs 2 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --reset --decay 600-1200-1800-2400-3000 --lr_patch_size --data_range 1-800 --data_train DIV2K

#CUDA_VISIBLE_DEVICES=6,7 python3.7 main.py --scale 2+3+4 --test_scale 2+3+4 --save RDN_MSL_p32_BIx234_DIV2K --model RDN_MSL --epochs 3000 --batch_size 16 --patch_size 32 --n_GPUs 2 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --reset --decay 600-1200-1800-2400-3000 --lr_patch_size --data_range 1-800 --data_train DIV2K

#CUDA_VISIBLE_DEVICES=6 python3.7 main.py --scale 2+3+4 --test_scale 2+3+4 --save RDN_MSL_p32_BIx234_DIV2K --model RDN_MSL --epochs 3000 --batch_size 16 --patch_size 32 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --reset --decay 600-1200-1800-2400-3000 --lr_patch_size --data_range 1-800 --data_train DIV2K

#CUDA_VISIBLE_DEVICES=6 python3.7 main.py --scale 2+3+4 --test_scale 2+3+4 --save RDN_MSL_p32_BIx234_DIV2K --model RDN_MSL --epochs 3000 --batch_size 16 --patch_size 32 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --resume -1 --decay 600-1200-1800-2400-3000 --lr_patch_size --data_range 1-800 --data_train DIV2K --load RDN_MSL_p32_BIx234_DIV2K

#CUDA_VISIBLE_DEVICES=6 python3.7 main.py --scale 2+3+4 --test_scale 2+3+4 --save RDN_MSL_p32_BIx234_DIV2K_latest --model RDN_MSL --epochs 3000 --batch_size 16 --patch_size 32 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5+Set14+B100+Urban100+Manga109 --decay 600-1200-1800-2400-3000 --lr_patch_size --data_range 1-800 --data_train DIV2K --pre_train ../experiment/RDN_MSL_p32_BIx234_DIV2K/model/model_latest.pt --test_only --save_results

#CUDA_VISIBLE_DEVICES=7 python3.7 main.py --scale 2 --test_scale 2 --save RDN_MSL_p32_BIx2_DIV2K --model RDN_MSL --epochs 1000 --batch_size 16 --patch_size 32 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --decay 200-400-600-800-1000 --lr_patch_size --data_range 1-800 --data_train DIV2K

#CUDA_VISIBLE_DEVICES=7 python3.7 main.py --scale 2 --test_scale 2 --save RDN_MSL_p32_BIx2_DIV2K --model RDN_MSL --epochs 1000 --batch_size 16 --patch_size 32 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --decay 200-400-600-800-1000 --lr_patch_size --data_range 1-800 --data_train DIV2K --load RDN_MSL_p32_BIx2_DIV2K --resume -1

#CUDA_VISIBLE_DEVICES=7 python3.7 main.py --scale 2 --test_scale 2 --save RDN_MSL_p32_BIx2_DIV2K_latest --model RDN_MSL --epochs 1000 --batch_size 16 --patch_size 32 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5+Set14+B100+Urban100+Manga109 --decay 200-400-600-800-1000 --lr_patch_size --data_range 1-800 --data_train DIV2K --pre_train ../experiment/RDN_MSL_p32_BIx2_DIV2K/model/model_latest.pt --test_only --save_results

#CUDA_VISIBLE_DEVICES=7 python3.7 main.py --scale 3 --test_scale 3 --save RDN_MSL_p32_BIx3_DIV2K --model RDN_MSL --epochs 1000 --batch_size 16 --patch_size 32 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --decay 200-400-600-800-1000 --lr_patch_size --data_range 1-800 --data_train DIV2K --pre_train ../experiment/RDN_MSL_p32_BIx2_DIV2K/model/model_latest.pt

#CUDA_VISIBLE_DEVICES=7 python3.7 main.py --scale 3 --test_scale 3 --save RDN_MSL_p32_BIx3_DIV2K --model RDN_MSL --epochs 1000 --batch_size 16 --patch_size 32 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --decay 200-400-600-800-1000 --lr_patch_size --data_range 1-800 --data_train DIV2K --load RDN_MSL_p32_BIx3_DIV2K --resume -1

#CUDA_VISIBLE_DEVICES=7 python3.7 main.py --scale 3 --test_scale 3 --save RDN_MSL_p32_BIx3_DIV2K_latest --model RDN_MSL --epochs 1000 --batch_size 16 --patch_size 32 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5+Set14+B100+Urban100+Manga109 --decay 200-400-600-800-1000 --lr_patch_size --data_range 1-800 --data_train DIV2K --pre_train ../experiment/RDN_MSL_p32_BIx3_DIV2K/model/model_latest.pt --test_only --save_results

#CUDA_VISIBLE_DEVICES=7 python3.7 main.py --scale 4 --test_scale 4 --save RDN_MSL_p32_BIx4_DIV2K --model RDN_MSL --epochs 1000 --batch_size 16 --patch_size 32 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --decay 200-400-600-800-1000 --lr_patch_size --data_range 1-800 --data_train DIV2K --pre_train ../experiment/RDN_MSL_p32_BIx2_DIV2K/model/model_latest.pt --reset

#CUDA_VISIBLE_DEVICES=7 python3.7 main.py --scale 4 --test_scale 4 --save RDN_MSL_p32_BIx4_DIV2K --model RDN_MSL --epochs 1000 --batch_size 16 --patch_size 32 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --decay 200-400-600-800-1000 --lr_patch_size --data_range 1-800 --data_train DIV2K --load RDN_MSL_p32_BIx4_DIV2K --resume -1

#CUDA_VISIBLE_DEVICES=7 python3.7 main.py --scale 4 --test_scale 4 --save RDN_MSL_p32_BIx4_DIV2K_latest --model RDN_MSL --epochs 1000 --batch_size 16 --patch_size 32 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5+Set14+B100+Urban100+Manga109 --decay 200-400-600-800-1000 --lr_patch_size --data_range 1-800 --data_train DIV2K --pre_train ../experiment/RDN_MSL_p32_BIx4_DIV2K/model/model_latest.pt --test_only --save_results


#CUDA_VISIBLE_DEVICES=6,7 python3.7 main.py --scale 2+3+4 --test_scale 2+3+4 --save RCAN_MSL_BIx234_DIV2K --model RCAN_MSL --epochs 3000 --batch_size 16 --patch_size 48 --n_GPUs 2 --n_threads 16 --ext sep --data_test Set5 --reset --decay 600-1200-1800-2400-3000 --lr_patch_size --data_range 1-800 --data_train DIV2K --n_resgroups 10 --n_resblocks 20 --n_feats 64 



