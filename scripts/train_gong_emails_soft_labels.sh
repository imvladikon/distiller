cd ..
export ROOT_DIR=$(pwd)
#export WANDB_DISABLED=true
mkdir -p $ROOT_DIR/models/gong_soft_labels
cd $ROOT_DIR/models/gong_soft_labels &&
  python3 $ROOT_DIR/train_eval_multilabel_model.py \
    --model_name=bert-base-uncased \
    --dataset_config=gong_soft_labels \
    --max_seq_length=512 \
    --do_train \
    --do_eval \
    --do_lower_case \
    --one_cycle_train \
    --tokenizer_name=bert-base-uncased \
    --output_dir=$ROOT_DIR/models/gong_soft_labels/bert-base-uncased-ft \
    --train_batch_size=24 \
    --val_batch_size=12 \
    --gradient_accumulation_steps=2 \
    --learning_rate=3e-5 \
    --num_train_epochs=5 \
    --seed=42 \
    --calculate_per_class \
    --threshold=0.5
#   it's metric evaluation threshold. threshold for soft labels is in dataset config
