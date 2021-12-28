  CUDA_VISIBLE_DEVICES=0 python test.py \
  --image-path ../task1_input \
  --box-path ../task1_output \
  --output-path task2_output \
  --trained-model ../../SRN_best_0.778.pth \
  --height 32 \
  --width 200 \
  --voc_type ALLCASES_SYMBOLS \
  --max_len 800
