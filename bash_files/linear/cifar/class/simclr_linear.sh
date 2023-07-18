python3 main_linear.py \
  --dataset cifar100 \
  --encoder resnet18 \
  --data_dir $DATA_DIR \
  --split_strategy class \
  --num_tasks 5 \
  --max_epochs 100 \
  --gpus 0 \
  --precision 16 \
  --optimizer sgd \
  --scheduler step \
  --lr 1.0 \
  --lr_decay_steps 60 80 \
  --weight_decay 0 \
  --batch_size 256 \
  --num_workers 7 \
  --name simclr-cifar100-5T-linear-eval-t0 \
  --pretrained_feature_extractor $PRETRAINED_PATH \
  --project semi-cassle-linear \
  --entity pigpeppa \
  --wandb \
  --save_checkpoint
