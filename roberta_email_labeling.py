import logging
import os
import torch

from modeling.gong import bert_seq_classification as bsc
from transformers import RobertaTokenizer
from modeling.gong.bert_seq_classification import InputExample
from misc import dotdict

from pathlib import Path
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

ROBERTA_PRETRAINED_PATH = Path(r'./models/roberta-base')

# ## INITIAL SET UP

#from transformers import RobertaModel, RobertaTokenizer
#if not ROBERTA_PRETRAINED_PATH.exists():
#     ROBERTA_PRETRAINED_PATH.resolve().mkdir(parents=True)
#model = RobertaModel.from_pretrained('roberta-base')            # download from Internet
#model.save_pretrained(ROBERTA_PRETRAINED_PATH)                       # save
#tokenizer = RobertaTokenizer.from_pretrained('roberta-base')    # download from Internet
#tokenizer.save_pretrained(ROBERTA_PRETRAINED_PATH)                   # save
#exit()


TRAIN_LABELS = ["Reject", "Scheduling", "Pricing", 'PI', "OOO", "Payments", "Contract", "reply"]


# TRAIN_LABELS = ["Reject", "Scheduling", "Pricing", 'PI', "OOO"]
# TRAIN_LABELS = ["Reject", "Scheduling", "Pricing", 'PI']
# TRAIN_LABELS = ["Reject", "Scheduling", "Pricing", ]
# TRAIN_LABELS = ["Reject", "OOO"]
# TRAIN_LABELS = ["PI"]
# TRAIN_LABELS = [ "Pricing", "Payments", "Contract"]
TEST_LABELS = TRAIN_LABELS


DATA_PATH = Path(r'./gong_data/0')

model_name = 'final_model_8_tag_case_sensitive'
suffix = '_512x'



ROBERTA_FINETUNED_PATH = Path(f'./models/tuned/tuned_{model_name}')

CLAS_DATA_PATH = DATA_PATH / 'class/output'
CLAS_DATA_PATH.mkdir(exist_ok=True, parents=True)

args = {
    "do_train": False,
    "do_eval": True,
    "one_cycle_train": True,
    "train_format_with_proba": False,
    "input_fname": 'train.csv',
    #"input_train_fname":'training_set_weak_labels.csv',
    #"input_test_fname":'training_set_weak_labels.csv',
    # "input_fname": 'train_augmented.csv',
    "train_size": -1,
    "val_size": -1,
    "data_dir": DATA_PATH,
    "bert_tokenizer": Path(r'./models/roberta-base'),
    "bert_model_dir": ROBERTA_PRETRAINED_PATH,
    "output_model_dir": ROBERTA_FINETUNED_PATH,

    "tuned_model_dir": ROBERTA_FINETUNED_PATH,

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


#endregion

if __name__ == '__main__':
    bsc.set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
    n_gpu = torch.cuda.device_count()
    args.n_gpu = n_gpu

    if 'n_threads' in args:
        torch.set_num_threads(args['n_threads'])
        logger.info(f"Setting #threads to {args['n_threads']}")

    # Train
    if args.do_train:
        logger.info('Training')
        logger.info(f"device: {device} \t #gpu: {n_gpu}")
        logger.info(f'Using model: {str(args.bert_model_dir)}')

        bsc.set_seed(args.seed)
        tokenizer = RobertaTokenizer.from_pretrained(os.path.abspath(args.bert_tokenizer.resolve().as_posix()))
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

        model = bsc.RobertaForMultiLabelSequenceClassification.from_pretrained(args.bert_model_dir,
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
        model.unfreeze_bert_encoder(['pooler', '11', '10', '9', '8', '7', '6', '5'])  # , '9', '8', '7', '6'])
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

    if args.do_eval:
        logger.info('Testing')
        #args.input_fname = args.input_test_fname

        processor = bsc.BinaryLabelTextProcessor(TRAIN_LABELS)

        tokenizer = RobertaTokenizer.from_pretrained(args.bert_tokenizer.resolve().as_posix())

        label_list = processor.get_labels()
        num_labels = len(label_list)

        args.eval_fname = 'test.csv'

        # args.input_fname = '83k_100perCompany.csv'

        eval_examples = processor.get_dev_examples(args)
        eval_features = bsc.convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)

        model = bsc.RobertaForMultiLabelSequenceClassification.from_pretrained(args.output_model_dir.resolve().as_posix(),
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
        df = pd.read_csv(DATA_PATH / args.eval_fname)
        if 'label_text' not in df.columns:
            df['label_text'] = 'no label'
            df['label_text'] = df['label_text'].apply(lambda x: [x])
        gdf = df.groupby('document_id').agg({'label_text': 'sum', 'document_text': 'first', }).reset_index()
        gdf['hash'] = gdf.document_id.apply(lambda x: hash(str(x)))
        df = pd.merge(gdf, res, left_on='hash', right_on='guids')
        print(f'Shape of resulting dataframe: {df.shape}')

        # sel = df[(df.iloc[:, 2:9] < 0.01).all(axis=1) | (df.iloc[:, 2:9] > 0.65).any(axis=1)]
        # sel['label_text'] = sel.iloc[:, 2:9].apply(
        #     lambda x: [f'not {it[0].split("_")[1]}' if it[1] < 0.5 else it[0].split("_")[1] for it in x.items()],
        #     axis=1)
        # sel['label_text'] = sel['label_text'].apply(set)
        # sel.to_csv(args['data_dir'] / 'augmented.csv', index=False)

        calibrated = None
        calibrated_fname = os.path.join(args.tuned_model_dir, 'calibration.pckl')
        if os.path.exists(calibrated_fname):
            calibrated = pd.read_pickle(calibrated_fname)
            logger.info('Loading score calibrations')
            for lbl in TRAIN_LABELS:
                lbl_proba = f'proba_{lbl.lower()}'
                df[lbl_proba] = np.interp(df[lbl_proba], calibrated['x'], calibrated[lbl])
        else:
            logger.warning(f'Score calibrations file "{calibrated_fname}" was not found')

        df = df.drop(['hash', 'guids'], axis=1)
        cols = df.columns.tolist()
        cols.remove('document_text')
        cols += ['document_text']
        df = df[cols]
        df.to_csv(args['data_output_dir'] / f'{args.input_fname.split(".")[0]}_{model_name}_.csv', index=False)

        # df = join_labels(df)
        # df.to_csv(DATA_PATH / '5_unlbl25k_train.csv', index=False)

        #
        # from matplotlib import pyplot as plt
        #
        # style = ['b-', 'r-', 'm-', 'c-', 'g-', 'k-', 'b:', 'r:', 'm:', 'c:', 'g:', ]
        # plt.figure()
        #
        # recall_threshold = 0.85
        # precision_threshold = 0.9
        # for ix, label in enumerate(TRAIN_LABELS):
        #     label = label.lower()
        #     precision, recall, thresholds = precision_recall_curve(df[f"true_{label}"].values, df[f'proba_{label}'].values)
        #     thresh = thresholds[np.diff(recall<=recall_threshold).argmax()]
        #     # thresh = thresholds[np.diff(precision<=precision_threshold).argmax()]
        #     predicted_class = (df[f"proba_{label}"]>=thresh).values.astype(int)
        #     print(f'{label} : {thresh}')
        #     print(f'Avg Prc: {label:12} {average_precision_score(df[f"true_{label}"].values, predicted_class)}')
        #     print(f'ROC AUC: {label:12} {roc_auc_score(df[f"true_{label}"].values, predicted_class)}')
        #     print(classification_report(df[f"true_{label}"].values, predicted_class, digits=5, target_names=['no '+label, label]))
        #
        #     plt.plot(recall, precision, style[ix], label=label)
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.xlim((0.5, 1))
        # plt.ylim((0.5, 1))
        # plt.title(f'ROBERTa 5-11 {model_name}{suffix} ({args.input_fname.split(".")[0]})')
        # plt.grid(True)
        # plt.legend()
        # plt.savefig(f'roberta_{"".join(TRAIN_LABELS)}_{args.input_fname.split(".")[0]}_{model_name}{suffix}.png')
        # # plt.show()

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
        plt.title(f'ROBERTa 5-11 {model_name}{suffix} ({args.eval_fname.split(".")[0]})')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'roberta_{"".join(TRAIN_LABELS)}_{args.eval_fname.split(".")[0]}_{model_name}.png')
        # plt.show()


    print(f'Model {model_name}{suffix}')
