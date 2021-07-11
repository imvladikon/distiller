import matplotlib

from const import *
from utils import set_seed
from utils.dataloader import load_dataset
from datasets import concatenate_datasets

matplotlib.use("agg")

import logging
import os
import torch

from modeling.gong import bert_seq_classification as bsc
from modeling.bert_multilabel_classification import BertForMultiLabelSequenceClassification
from modeling.bert_cnn_classification import BertForClassificationCNN

from transformers import AutoTokenizer
from collections import Counter

import pandas as pd
import numpy as np

from sklearn.metrics import precision_recall_curve, classification_report, roc_auc_score, average_precision_score

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 100)


def report(some_set, set_name):
    print(f'\n\n{"=" * 10}  {set_name} \n')
    c = Counter()
    for row in some_set.label_text.tolist():
        c.update(list(row))
    for k, v in c.most_common(20):
        print(f'{k:15s}: {v}')
    print(f'\tTotal: {some_set.shape[0]}')


def join_labels(df):
    for ix, row in df.iterrows():
        for lbl in row['label_text']:
            if ' ' in lbl:
                df.loc[ix, 'proba_' + lbl.split()[-1]] = 0
            else:
                df.loc[ix, 'proba_' + lbl] = 1
        if ix < 5:
            print(row)
    return df


if __name__ == '__main__':
    # TODO: add argparse
    from config import default

    args = default.args
    set_seed(args.seed)
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

    ds = load_dataset(
        train_filename=str(ROOT_DIR / "data" / "0" / "train.csv"),
        val_filename=str(ROOT_DIR / "data" / "0" / "test.csv"),
        tokenizer=tokenizer,
        max_seq_length=512,
        train_format_with_proba=args.train_format_with_proba,
        train_size=args.train_size,
        val_size=args.val_size
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
        ones = all_proba.numpy()
        res = pd.concat([
            pd.DataFrame(guids, columns=["guids"]),
            pd.DataFrame(ones, columns=['proba_' + el.lower() for el in label_list]),
            pd.DataFrame(true_class, columns=['true_' + el.lower() for el in label_list])
        ], axis=1)

        agg_by = {'proba_' + el.lower(): 'max' for el in label_list}
        agg_by.update({'true_' + el.lower(): 'max' for el in label_list})
        res = res.groupby('guids').agg(agg_by).reset_index()
        df = pd.read_csv(ROOT_DIR / 'data' / '0' / 'test.csv')
        if 'label_text' not in df.columns:
            df['label_text'] = 'no label'
            df['label_text'] = df['label_text'].apply(lambda x: [x])
        if 'subject' in df:
            gdf = df.groupby('document_id').agg({'label_text': 'sum', 'subject': 'first', 'company_id': 'first',
                                                 'document_text': 'first'}).reset_index()
        else:
            gdf = df.groupby('document_id').agg({'label_text': 'sum', 'document_text': 'first'}).reset_index()

        gdf['hash'] = gdf.document_id.apply(lambda x: hash(str(x)))
        df = pd.merge(gdf, res, left_on='hash', right_on='guids')
        print(f'Shape of resulting dataframe: {df.shape}')

        df = df.drop(['hash', 'guids'], axis=1)
        cols = df.columns.tolist()
        cols.remove('document_text')
        cols += ['document_text']
        df = df[cols]

        calibrated_fname = os.path.join(args.output_model_dir, 'calibration.pckl')
        if os.path.exists(calibrated_fname):
            calibrated = pd.read_pickle(calibrated_fname)
            logger.info('Loading score calibrations')
            if all(it in calibrated.columns for it in label_list):
                for lbl in label_list:
                    df['proba_' + lbl.lower()] = np.interp(df['proba_' + lbl.lower()], calibrated['x'], calibrated[lbl])
        else:
            logger.warning(f'No calibrations file "{calibrated_fname}" was found')

        # df.to_csv(args['data_output_dir'] / f'{args.input_fname.split(".")[0]}_{model_name}{suffix}.csv', index=True)

        # df = join_labels(df)
        # df.to_csv(DATA_PATH / '5_unlbl25k_train.csv', index=False)

        from matplotlib import pyplot as plt

        style = ['b-', 'r-', 'm-', 'c-', 'g-', 'k-', 'b:', 'r:', 'm:', 'c:', 'g:', ]
        plt.figure()

        recall_threshold = 0.9
        precision_threshold = 0.9
        for ix, label in enumerate(label_list):
            label = label.lower()
            df_chk = df[df.label_text.str.contains(label)]
            precision, recall, thresholds = precision_recall_curve(df_chk[f"true_{label}"].values,
                                                                   df_chk[f'proba_{label}'].values)
            thresh = thresholds[np.diff(recall <= recall_threshold).argmax()]
            # thresh = thresholds[np.diff(precision<=precision_threshold).argmax()]
            predicted_class = (df_chk[f"proba_{label}"] >= thresh).values.astype(int)
            print(f'{label} : {thresh}')
            print(f'Avg Prc: {label:12} {average_precision_score(df_chk[f"true_{label}"].values, predicted_class)}')
            print(f'ROC AUC: {label:12} {roc_auc_score(df_chk[f"true_{label}"].values, predicted_class)}')
            print(classification_report(df_chk[f"true_{label}"].values, predicted_class, digits=5,
                                        target_names=['no ' + label, label]))

            plt.plot(recall, precision, style[ix], label=label)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim((0.5, 1))
        plt.ylim((0.5, 1))
        plt.title(f'BERT 5-11 {model_name} ({args.input_fname.split(".")[0]})')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{args.eval_fname.split(".")[0]}_{model_name}.png')
        plt.show()
