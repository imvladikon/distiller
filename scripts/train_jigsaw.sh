export ROOT_DIR=".."
mkdir -p $ROOT_DIR/models/jigsaw
cd $ROOT_DIR/models/jigsaw &&
  python3 $ROOT_DIR/bert_train_model_multilabels.py \
    --model_name=bert-base-uncased \
    --dataset_config=jigsaw \
    --max_seq_length=512 \
    --do_train \
    --do_eval \
    --do_lower_case \
    --one_cycle_train \
    --tokenizer_name=bert-base-uncased \
    --output_dir=$ROOT_DIR/models/jigsaw/bert-base-uncased-ft \
    --train_batch_size=24 \
    --val_batch_size=12 \
    --gradient_accumulation_steps=2 \
    --learning_rate=3e-5 \
    --num_train_epochs=5 \
    --seed=42 \
    --threshold=0.5
