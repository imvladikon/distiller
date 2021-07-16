import matplotlib

from const import *
from utils import set_seed
from utils.dataloader import load_dataset, read_data

matplotlib.use("agg")

import logging
import torch

from modeling.gong import bert_seq_classification as bsc
from modeling.bert_multilabel_classification import BertForMultiLabelSequenceClassification
from modeling.bert_cnn_classification import BertForClassificationCNN

from transformers import AutoTokenizer


logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)



if __name__ == '__main__':
    # TODO: add argparse
    from config import default

    args = default.args
    set_seed(args.seed)
    labels = [
        'toxic', 'severe_toxic', 'obscene', 'threat',
        'insult', 'identity_hate'
    ]
    text_column = "comment_text"

    id2label = {k: v for k, v in enumerate(labels)}
    label2id = {v: k for k, v in id2label.items()}
    label_list = labels

    model_name = args.model_name

    if 'n_threads' in args:
        torch.set_num_threads(args['n_threads'])
        logger.info(f"Setting #threads to {args['n_threads']}")

    logger.info(f"device: {device} \t #number of gpu: {args.n_gpu}")
    logger.info(f'Using model: {str(args.bert_model_dir)}')

    tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer)
    model = BertForMultiLabelSequenceClassification.from_pretrained(args.bert_model_dir,
                                                                    num_labels=len(label_list)).to(device)
    # model.resize_token_embeddings(len(tokenizer))

    ds = read_data(
        train_filename=str(ROOT_DIR / "data" / "jigsaw" / "train.csv"),
        tokenizer=tokenizer,
        max_seq_length=512,
        train_size=args.train_size,
        val_size=args.val_size,
        map_label_columns = dict(zip(labels,labels)), # identical
        text_column = "comment_text",
        val_split_size=0.2,
        print_label_dist = True
    )

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



