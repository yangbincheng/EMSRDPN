CUDA_VISIBLE_DEVICES=0 python3.7 main.py --scale 4 --test_scale 4 --save EMSRDPN_ResNet_BIx4 --model EMSRDPN --epochs 1250 --batch_size 16 --patch_size 48 --n_GPUs 1 --n_threads 16 --SRDPNconfig B --ext sep --data_test Set5 --reset --decay 250-500-750-1000-1250 --lr_patch_size --data_range 1-3450 --data_train Flickr2K

CUDA_VISIBLE_DEVICES=0 python3.7 main.py --scale 4 --test_scale 4 --save EMSRDPN_DenseNet_BIx4 --model EMSRDPN --epochs 1250 --batch_size 16 --patch_size 48 --n_GPUs 1 --n_threads 16 --SRDPNconfig C --ext sep --data_test Set5 --reset --decay 250-500-750-1000-1250 --lr_patch_size --data_range 1-3450 --data_train Flickr2K

#CUDA_VISIBLE_DEVICES=1 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348_DIV2K --model EMSRDPN --epochs 4000 --batch_size 16 --patch_size 48 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --reset --decay 800-1600-2400-3200-4000 --lr_patch_size --data_range 1-800 --data_train DIV2K

#CUDA_VISIBLE_DEVICES=0,1 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348 --model EMSRDPN --epochs 5000 --batch_size 16 --patch_size 48 --n_GPUs 2 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --reset --decay 1000-2000-3000-4000-5000 --lr_patch_size --data_range 1-3450 --data_train Flickr2K

#CUDA_VISIBLE_DEVICES=0,1 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348 --model EMSRDPN --epochs 5000 --batch_size 16 --patch_size 48 --n_GPUs 2 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --resume -1 --decay 1000-2000-3000-4000-5000 --lr_patch_size --data_range 1-3450 --data_train Flickr2K --load EMSRDPN_BIx2348

#CUDA_VISIBLE_DEVICES=0 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348_test --model EMSRDPN --epochs 5000 --batch_size 16 --patch_size 48 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5+Set14+B100+Urban100+Manga109 --reset --decay 1000-2000-3000-4000-5000 --lr_patch_size --data_range 1-3450 --data_train Flickr2K --pre_train ../experiment/EMSRDPN_BIx2348/model/model_latest.pt --test_only --save_results

#CUDA_VISIBLE_DEVICES=0 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348_test+ --model EMSRDPN --epochs 5000 --batch_size 16 --patch_size 48 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5+Set14+B100+Urban100+Manga109 --reset --decay 1000-2000-3000-4000-5000 --lr_patch_size --data_range 1-3450 --data_train Flickr2K --pre_train ../experiment/EMSRDPN_BIx2348/model/model_latest.pt --test_only --save_results --self_ensemble

#CUDA_VISIBLE_DEVICES=0 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348_test_multi_scale_infer --model EMSRDPN --epochs 5000 --batch_size 16 --patch_size 48 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --reset --decay 1000-2000-3000-4000-5000 --lr_patch_size --data_range 1-3450 --data_train Flickr2K --pre_train ../experiment/EMSRDPN_BIx2348/model/model_latest.pt --test_only --save_results --multi_scale_infer

