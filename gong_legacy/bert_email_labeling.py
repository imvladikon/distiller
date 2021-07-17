import matplotlib
matplotlib.use("agg")

import logging
import os
import ast
import torch

from gong_utils.nlp.bert import gong_bert_seq_classification as bsc
from gong_utils.nlp.bert.gong_bert_seq_classification  import InputExample
from gong_utils.misc.dotdict import dotdict
from collections import Counter

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report, roc_auc_score, average_precision_score

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 100)

BERT_PRETRAINED_PATH = Path(r'./models/bert-base-uncased')

# ## INITIAL SET UP

#from transformers import BertModel, BertTokenizer
#if not BERT_PRETRAINED_PATH.exists():
#    BERT_PRETRAINED_PATH.resolve().mkdir(parents=True) 
#model = BertModel.from_pretrained('bert-base-uncased')            # download from Internet
#model.save_pretrained(BERT_PRETRAINED_PATH)                       # save
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')    # download from Internet
#tokenizer.save_pretrained(BERT_PRETRAINED_PATH)                   # save
#exit()

TRAIN_LABELS = ["Reject", "Scheduling", "Pricing", 'PI', "OOO", "Payments", "Contract", "reply"]
# TRAIN_LABELS = ["Reject", "Scheduling", "Pricing", 'PI', "OOO"]
# TRAIN_LABELS = ["Reject", "Scheduling", "Pricing", 'PI']
# TRAIN_LABELS = ["Reject", "Scheduling", "Pricing", ]
# TRAIN_LABELS = ["Reject", "OOO"]
# TRAIN_LABELS = ["Reject"]
# TRAIN_LABELS = ["Scheduling"]
# TRAIN_LABELS = ["Reject"]
# TRAIN_LABELS = ["Contract"]
# TRAIN_LABELS = ["PI"]
# TRAIN_LABELS = [ "Reject", "Scheduling", "Pricing", "PI", "Payments", "Contract"]
# TRAIN_LABELS = ["Reject", "Scheduling", "Pricing", 'OOO', "Payments", "Contract"]
TEST_LABELS = TRAIN_LABELS

model_name = 'reply'
suffix = '_x512'

DATA_PATH = Path(r'./gong_data/0')
BERT_FINETUNED_PATH = Path(f'./models/tuned/tuned_bert{model_name}')

CLAS_DATA_PATH = DATA_PATH/'class/output'
CLAS_DATA_PATH.mkdir(exist_ok=True, parents=True)

args = {
    "do_train": False,
    "do_eval": True,
    "one_cycle_train": True,
    "train_format_with_proba": False,
    # "input_fname": 'train_sel+cur+aug.csv',
    "train_size": -1,
    "val_size": -1,
    "data_dir": DATA_PATH,
    "task_name": "email_reject",
    "bert_tokenizer": Path(r'./models/bert-base-uncased'),
    "bert_model_dir": BERT_PRETRAINED_PATH,
    "output_model_dir": BERT_FINETUNED_PATH,
    "data_output_dir": CLAS_DATA_PATH,
    "max_seq_length": 512,
    "do_lower_case": True,
    "train_batch_size": 24,
    "eval_batch_size": 12,
    "n_threads": 4,
    "learning_rate": 3e-5,
    "num_train_epochs": 3,
    "warmup_linear": False,
    "warmup_proportion": 0.1,
    "no_cuda": False,
    "local_rank": -1,
    "seed": 888,
    "gradient_accumulation_steps": 1,
    "optimize_on_cpu": True,
    "fp16": False,
    "loss_scale": 128
}
args = dotdict(args)



def report(some_set, set_name):
    print(f'\n\n{"="*10}  {set_name} \n')
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
    bsc.set_seed(args.seed)

    # download_data_from_s3(True)
    # parquet2csv(TRAIN_LABELS)
    # parquet2csv_combined(TRAIN_LABELS)

    device = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
    n_gpu = torch.cuda.device_count()
    args.n_gpu = n_gpu

    if 'n_threads' in args:
        torch.set_num_threads(args['n_threads'])
        logger.info(f"Setting #threads to {args['n_threads']}")


    # model_base = model_name
    # for ix in range(7):
    #     args.input_fname = f'train_cv{ix}.csv'
    #     args.output_model_dir = Path(BERT_FINETUNED_PATH.as_posix() + f'_cv{ix}')
    #     model_name = f'{model_base}_cv{ix}'

    # Train
    if args.do_train:
        logger.info('Training')
        logger.info(f"device: {device} \t #gpu: {n_gpu}")
        logger.info(f'Using model: {str(args.bert_model_dir)}')

        bsc.set_seed(args.seed)
        tokenizer = bsc.BertTokenizer.from_pretrained(args.bert_tokenizer.resolve().as_posix())
        processor = bsc.BinaryLabelTextProcessor(TRAIN_LABELS)

        label_list = processor.get_labels()

        if args.train_format_with_proba:
            data_df = pd.read_csv(os.path.join(args.data_dir, args.input_fname))
            lbls = ['proba_' + el.lower() for el in TRAIN_LABELS]
            train_examples = data_df.apply(
                lambda x: InputExample(guid=hash(x['document_id']),
                                       text_a=x['document_text'],
                                       labels=x[lbls].tolist()),
                axis=1).tolist()
        else:
            train_examples = processor.get_train_examples(args, size=args.train_size)

        train_features = bsc.convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)

        model = bsc.BertForMultiLabelSequenceClassification.from_pretrained(args.bert_model_dir,
                                                                            num_labels=len(label_list))
        # model.resize_token_embeddings(len(tokenizer))
        model.to(device)

        if args.one_cycle_train:
            model.unfreeze_bert_encoder(['pooler'])
            prev_num_train_epochs = args.num_train_epochs
            args.num_train_epochs = 1
            bsc.train(args, train_features, model, device)
            args.num_train_epochs = prev_num_train_epochs

        # args.num_train_epochs = prev_num_train_epochs
        model.unfreeze_bert_encoder(['pooler', '11', '10', '9', '8', '7', '6', '5'])  #  , '9', '8', '7', '6'])
        global_step, tr_loss = bsc.train(args, train_features, model, device)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Save a trained model
        if not args.output_model_dir.exists():
            args.output_model_dir.resolve().mkdir(parents=True)
        model.save_pretrained(args.output_model_dir)

        # print('\n\n **** Evaluating ***\n')
        # args.input_fname = 'dev.csv'
        # eval_examples = processor.get_dev_examples(args)
        # eval_features = bsc.convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)

        # logits, true_class, guids = bsc.predict(args, eval_features, model, device)
        #
        # all_proba = torch.sigmoid(torch.Tensor(logits))
        # thresh = 0.8
        # predicted_class = (all_proba >= thresh).numpy().astype(float)
        # print(classification_report(true_class, predicted_class, digits=5, target_names=label_list))

    # args.input_fname = f'30_long.csv'
    # args.input_fname = f'test_cv{ix}.csv'
    # args.input_fname = f'test_cv6.csv'
    args.input_fname = f'test.csv'
    # args.input_fname = f'unlbl_50k.csv'
    # args.input_fname = f'dev.csv'
    # args.input_fname = f'train.csv'
    if args.do_eval:
        logger.info('Testing')

        processor = bsc.BinaryLabelTextProcessor(TRAIN_LABELS)

        tokenizer = bsc.BertTokenizer.from_pretrained(args.bert_tokenizer.resolve().as_posix())

        label_list = processor.get_labels()
        num_labels = len(label_list)

        args.eval_fname = 'test.csv'
        eval_examples = processor.get_dev_examples(args)
        eval_features = bsc.convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, cut_long=True)

        model = bsc.BertForMultiLabelSequenceClassification.from_pretrained(args.output_model_dir.resolve().as_posix(),
                                                                            num_labels=len(processor.get_labels()))
        model.to(device)

        logits, true_class, guids = bsc.predict(args, eval_features, model, device)

        all_proba = torch.sigmoid(torch.Tensor(logits))

        thresh = 0.8
        predicted_class = (all_proba >= thresh).numpy().astype(float)
        # ones = np.maximum(predicted_class, true_class)*0.9998 + 0.0001
        ones = all_proba.numpy()
        res = pd.concat([
            pd.DataFrame(guids, columns=["guids"]),
            pd.DataFrame(ones, columns=['proba_' + el.lower() for el in TRAIN_LABELS]),
            pd.DataFrame(true_class, columns=['true_' + el.lower() for el in TRAIN_LABELS])
        ], axis=1)

        agg_by = {'proba_' + el.lower(): 'max' for el in TRAIN_LABELS}
        agg_by.update({'true_' + el.lower(): 'max' for el in TRAIN_LABELS})
        res = res.groupby('guids').agg(agg_by).reset_index()

        df = pd.read_csv(DATA_PATH / args.input_fname)
        if 'label_text' not in df.columns:
            df['label_text'] = 'no label'
            df['label_text'] = df['label_text'].apply(lambda x: [x])
        if 'subject' in df:
            gdf = df.groupby('document_id').agg({'label_text': 'sum', 'subject':'first', 'company_id':'first', 'document_text': 'first'}).reset_index()
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
            if all(it in calibrated.columns for it in TRAIN_LABELS):
                for lbl in TRAIN_LABELS:
                    df['proba_' + lbl.lower()] = np.interp(df['proba_' + lbl.lower()], calibrated['x'], calibrated[lbl])
        else:
            logger.warning(f'No calibrations file "{calibrated_fname}" was found')

        df.to_csv(args['data_output_dir'] / f'{args.input_fname.split(".")[0]}_{model_name}{suffix}.csv', index=True)

        # df = join_labels(df)
        # df.to_csv(DATA_PATH / '5_unlbl25k_train.csv', index=False)


        from matplotlib import pyplot as plt
        style = ['b-', 'r-', 'm-', 'c-', 'g-', 'k-', 'b:', 'r:', 'm:', 'c:', 'g:', ]
        plt.figure()

        recall_threshold = 0.9
        precision_threshold = 0.9
        for ix, label in enumerate(TRAIN_LABELS):
            label = label.lower()
            df_chk = df[df.label_text.str.contains(label)]
            precision, recall, thresholds = precision_recall_curve(df_chk[f"true_{label}"].values, df_chk[f'proba_{label}'].values)
            thresh = thresholds[np.diff(recall<=recall_threshold).argmax()]
            # thresh = thresholds[np.diff(precision<=precision_threshold).argmax()]
            predicted_class = (df_chk[f"proba_{label}"]>=thresh).values.astype(int)
            print(f'{label} : {thresh}')
            print(f'Avg Prc: {label:12} {average_precision_score(df_chk[f"true_{label}"].values, predicted_class)}')
            print(f'ROC AUC: {label:12} {roc_auc_score(df_chk[f"true_{label}"].values, predicted_class)}')
            print(classification_report(df_chk[f"true_{label}"].values, predicted_class, digits=5, target_names=['no '+label, label]))

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

