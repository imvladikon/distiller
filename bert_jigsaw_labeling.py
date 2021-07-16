import argparse

import matplotlib

from config.datasets import DataFactory
from config.google_students_models import get_student_models
from const import *
from utils import set_seed, dotdict
from utils.dataloader import load_dataset, read_data

matplotlib.use("agg")

import logging
import torch

from modeling.gong import bert_seq_classification as bsc
from modeling.bert_multilabel_classification import BertForMultiLabelSequenceClassification
from modeling.bert_cnn_classification import BertForClassificationCNN

from transformers import (AutoTokenizer,
                          Trainer,
                          TrainerCallback,
                          AdamW,
                          get_linear_schedule_with_warmup)

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    ds = DataFactory.create_from_config(args.dataset_config,
                                        tokenizer=tokenizer,
                                        max_length=args.max_seq_length,
                                        train_size=args.train_size,
                                        val_size=args.val_size)
    dataset_info = ds.config
    ds = ds.dataset

    labels = dataset_info.labels
    id2label = dataset_info.id2label
    label2id = dataset_info.label2id
    label_list = dataset_info.labels

    if 'n_threads' in args:
        torch.set_num_threads(args['n_threads'])
        logger.info(f"Setting #threads to {args['n_threads']}")

    logger.info(f"device: {device} \t #number of gpu: {args.n_gpu}")
    logger.info(f'Using model: {str(args.model_name)}')

    model = BertForMultiLabelSequenceClassification.from_pretrained(args.model_name,
                                                                    num_labels=len(label_list)).to(device)
    # model.resize_token_embeddings(len(tokenizer))


    # loaders = datasets_as_loaders(ds, batch_size=64)

    train_features = ds["train"]
    eval_features = ds["test"]

    if args.do_train:
        logger.info('Training')

        if args.one_cycle_train:
            model.unfreeze_bert_encoder(['pooler'])
            prev_num_train_epochs = args.num_train_epochs
            args.num_train_epochs = 1
            bsc.train(args, train_features, model, device)
            args.num_train_epochs = prev_num_train_epochs

        # args.num_train_epochs = prev_num_train_epochs
        model.unfreeze_bert_encoder(['pooler', '11', '10', '9', '8', '7', '6', '5'])  # , '9', '8', '7', '6'])
        global_step, tr_loss = bsc.train(args, train_features, model, device)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        if not args.output_model_dir.exists():
            args.output_model_dir.resolve().mkdir(parents=True)

        model.save_pretrained(args.output_model_dir)

    if args.do_eval:
        logger.info('Testing')
        logits, true_class, guids = bsc.predict(eval_features, model, device, eval_batch_size=args.eval_batch_size)
        all_proba = torch.sigmoid(torch.Tensor(logits))
        thresh = 0.8
        predicted_class = (all_proba >= thresh).numpy().astype(float)
        # ones = np.maximum(predicted_class, true_class)*0.9998 + 0.0001


if __name__ == '__main__':
    hidden_size, num_layers = 256, 6
    student_model_name = get_student_models(hidden_size=hidden_size, num_layers=num_layers)

    parser = argparse.ArgumentParser(description='Jigsaw training and evaluation bert model')
    parser.add_argument("--model_name", default=student_model_name, type=str,
                        required=False)
    parser.add_argument("--dataset_config", default="jigsaw", type=str,
                        required=False)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true', default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--one_cycle_train", default=True, action='store_true', required=False)
    parser.add_argument("--train_size", default=-1, type=int, required=False)
    parser.add_argument("--val_size", default=-1, type=int, required=False)
    parser.add_argument("--tokenizer_name",
                        default="bert-base-uncased",
                        type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name",
                        required=False)
    parser.add_argument("--output_model_dir", default=str(ROOT_DIR / 'models' / 'distill'), type=str, required=False)
    parser.add_argument("--train_batch_size", default=24, type=int, required=False)
    parser.add_argument("--val_batch_size", default=12, type=int, required=False)
    parser.add_argument("--n_threads", default=4, type=int, required=False)
    parser.add_argument("--warmup_linear", default=0.1, type=float, required=False)
    parser.add_argument("--optimize_on_cpu", default=True, type=bool, required=False)
    parser.add_argument("--loss_scale", default=128, type=int, required=False)
    parser.add_argument("--labels_list", default=labels, type=list, required=False)
    parser.add_argument("--use_wandb", default=False, type=bool, required=False)
    parser.add_argument("--wandb_token", default='', type=str, required=False)
    parser.add_argument("--wandb_note", default='', type=str, required=False)
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
    parser.add_argument("--weight_decay", default=0.0, type=float,
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
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    args = parser.parse_args()
    set_seed(args.seed)
    args = vars(args)
    n_gpu = torch.cuda.device_count()
    args['n_gpu'] = n_gpu
    args = dotdict(args)
    main(args)
