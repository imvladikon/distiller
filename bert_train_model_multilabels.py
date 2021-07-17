import argparse

from transformers.modelcard import TrainingSummary

from config.datasets import DataFactory, DATASETS_CONFIG_INFO
from const import *
from metrics.multiclasseval import Multiclasseval
from metrics.utils import compute_multilabel_metrics
from utils import set_seed, dotdict
import logging
import torch
from functools import partial
from modeling.bert_multilabel_classification import BertForMultiLabelSequenceClassification
from modeling.bert_cnn_classification import BertForClassificationCNN

from transformers import (AutoTokenizer,
                          Trainer,
                          TrainerCallback,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          TrainingArguments,
                          PrinterCallback, BertPreTrainedModel)

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

AVAILABLE_CLASS_MODELS = {
    "BertForClassificationCNN": BertForClassificationCNN,
    "BertForMultiLabelSequenceClassification": BertForMultiLabelSequenceClassification
}


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)

    ds_with_info = DataFactory.create_from_config(args.dataset_config,
                                                  tokenizer=tokenizer,
                                                  max_length=args.max_seq_length,
                                                  train_size=args.train_size,
                                                  val_size=args.val_size)
    dataset_info = ds_with_info.config
    ds = ds_with_info.dataset

    id2label = dataset_info.id2label
    label2id = dataset_info.label2id
    label_list = dataset_info.labels

    if 'n_threads' in args:
        torch.set_num_threads(args['n_threads'])
        logger.info(f"Setting #threads to {args['n_threads']}")

    logger.info(f"device: {device} \t #number of gpu: {args.n_gpu}")

    model_class: BertPreTrainedModel = AVAILABLE_CLASS_MODELS.get(args.model_class,
                                                                  BertForMultiLabelSequenceClassification)
    logger.info(f'Using model class: {str(model_class)}')

    model = model_class.from_pretrained(args.model_name,
                                        num_labels=len(label_list)).to(device)
    # model.resize_token_embeddings(len(tokenizer))
    model.config.label2id = label2id
    model.config.id2label = id2label

    logger.info(f'Using model: {str(args.model_name)}')

    train_features = ds["train"]
    eval_features = ds["test"]

    metric = Multiclasseval()
    metric.threshold = args.threshold
    metric.num_classes = len(label_list)
    metric.labels = label_list
    metric.calculate_per_class = args.calculate_per_class
    compute_metrics = partial(compute_multilabel_metrics, metric=metric)

    optimizer = AdamW(model.parameters(),
                      lr=args.learning_rate,
                      eps=1e-8)

    total_steps = len(train_features) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    if args.use_wandb:
        wandb_env_vars = ["WANDB_NOTES", "WANDB_NAME", "WANDB_ENTITY", "WANDB_PROJECT", "WANDB_TAGS"]
        for v in wandb_env_vars:
            if v.lower() in args and args[v.lower()]:
                os.environ[v] = args[v.lower()]
    else:
        os.environ["WANDB_DISABLED"] = "true"

    if args.do_train:
        logger.info('Training')

        if args.one_cycle_train:
            model.unfreeze_bert_encoder(['pooler'])
            training_args = TrainingArguments(
                f"one_cycle_training",
                evaluation_strategy="epoch",
                learning_rate=args.learning_rate,
                per_device_train_batch_size=args.train_batch_size,
                per_device_eval_batch_size=args.val_batch_size,
                num_train_epochs=1,
                weight_decay=args.weight_decay,
                load_best_model_at_end=True,
                save_total_limit=2,
                report_to=None,
                gradient_accumulation_steps=args.gradient_accumulation_steps
            )
            trainer = Trainer(
                model,
                training_args,
                train_dataset=train_features,
                eval_dataset=eval_features,
                # data_collator=DataCollatorWithPadding(tokenizer),
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                callbacks=[PrinterCallback()],
                optimizers=(optimizer, scheduler)
            )
            trainer.train()
            args.num_train_epochs -= 1

        model.unfreeze_bert_encoder(['pooler', '11', '10', '9', '8', '7', '6', '5'])  # , '9', '8', '7', '6'])

    training_args = TrainingArguments(
        f"training",
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        save_total_limit=2,
        report_to=None,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_features,
        eval_dataset=eval_features,
        # data_collator=DataCollatorWithPadding(tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[PrinterCallback()],
        optimizers=(optimizer, scheduler)
    )

    if args.do_train:
        trainer.train()

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        trainer.save_model(args.output_dir)

        training_summary = TrainingSummary.from_trainer(
            trainer,
            language="en",
            license=license,
            model_name=args.model_name,
            finetuned_from="",
            tasks="multilabels classification",
            dataset=args.dataset_config,
        )
        model_card = training_summary.to_model_card()
        with open(os.path.join(args.output_dir, "README.md"), "w") as f:
            f.write(model_card)

    if args.do_eval:
        logger.info('evaluation')
        metrics = trainer.evaluate()

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        trainer.save_metrics("eval", metrics=metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training and evaluation bert model')

    parser.add_argument("--model_name", default="", type=str,
                        required=True)
    parser.add_argument("--dataset_config",
                        default="",
                        type=str,
                        required=True,
                        choices=list(DATASETS_CONFIG_INFO.keys()),
                        help="need to choose a dataset config")
    parser.add_argument("--model_class",
                        default="BertForMultiLabelSequenceClassification",
                        type=str,
                        required=False,
                        choices=list(AVAILABLE_CLASS_MODELS.keys()),
                        help="need to choose a model class. by default it's BertForMultiLabelSequenceClassification")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--calculate_per_class",
                        action='store_true',
                        help="Calculate metrics per class")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--one_cycle_train",
                        default=True,
                        action='store_true',
                        required=False)
    parser.add_argument("--train_size", default=-1, type=int, required=False)
    parser.add_argument("--val_size", default=-1, type=int, required=False)
    parser.add_argument("--tokenizer_name",
                        default="bert-base-uncased",
                        type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name",
                        required=False)

    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--train_batch_size", default=24, type=int, required=False)
    parser.add_argument("--val_batch_size", default=12, type=int, required=False)
    parser.add_argument("--n_threads", default=4, type=int, required=False)
    parser.add_argument("--warmup_linear", default=0.1, type=float, required=False)
    parser.add_argument("--optimize_on_cpu", default=True, type=bool, required=False)
    parser.add_argument("--loss_scale", default=128, type=int, required=False)

    parser.add_argument("--use_wandb", default=False, type=bool, required=False)
    parser.add_argument("--wandb_token", default='', type=str, required=False)
    parser.add_argument("--wandb_notes", default='', type=str, required=False)
    parser.add_argument("--wandb_project", default='', type=str, required=False)
    parser.add_argument("--wandb_entity", default='', type=str, required=False)
    parser.add_argument("--wandb_group", default='', type=str, required=False)
    parser.add_argument("--wandb_name", default='', type=str, required=False)
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=5, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="warmup_proportion")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--threshold", default=0.5, type=float, required=False)

    args = parser.parse_args()
    set_seed(args.seed)
    args = vars(args)
    n_gpu = torch.cuda.device_count()
    args['n_gpu'] = n_gpu
    args = dotdict(args)
    main(args)
