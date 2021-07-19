cd ..
export ROOT_DIR=$(pwd)
export student_model_name=google/bert_uncased_L-6_H-256_A-4
#export WANDB_DISABLED=true
mkdir -p $ROOT_DIR/models/jigsaw
cd $ROOT_DIR/models/jigsaw &&
  python3 $ROOT_DIR/train_distil_multilabel_model.py \
    --teacher_model_name=$ROOT_DIR/models/jigsaw/bert-base-uncased-ft \
    --student_model_name=$student_model_name \
    --dataset_config=jigsaw \
    --max_seq_length=512 \
    --do_train \
    --do_eval \
    --do_lower_case \
    --tokenizer_name=bert-base-uncased \
    --output_dir=$ROOT_DIR/models/jigsaw/distill/$student_model_name \
    --train_batch_size=24 \
    --val_batch_size=12 \
    --gradient_accumulation_steps=2 \
    --learning_rate=3e-5 \
    --num_train_epochs=25 \
    --seed=42 \
    --calculate_per_class \
    --temperature=1 \
    --kl_div_loss_weight=0.2 \
    --mse_loss_weight=0.1 \
    --task_loss_weight=0.5 \
    --emd_loss_weight=0.2 \
    --threshold=0.5