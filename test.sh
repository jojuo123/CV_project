  CUDA_VISIBLE_DEVICES=0 python eval.py \
  --test_data_dir ../drive/MyDrive/CV_project/dataset/val_set \
  --reuse_model ../drive/MyDrive/CV_project/SRN_best_0.778.pth \
  --lr 1e-4 \
  --workers 0 \
  --height 32 \
  --width 200 \
  --voc_type ALLCASES_SYMBOLS \
  --max_len 800 \
