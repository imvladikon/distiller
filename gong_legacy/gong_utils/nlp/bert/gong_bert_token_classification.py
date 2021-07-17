import logging
import os
import re
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

import random
import numpy as np
from transformers import AdamW
from transformers import BertTokenizer
from transformers import BertPreTrainedModel, BertModel
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def tokenize(text, tokenizer, label_list):
    words = text.split()
    raw_tokens = []
    # take care of punctuation marks inside words
    for w in words:
        toks = tokenizer.tokenize(w)
        for ix in range(len(toks)-1, 0, -1):
            if not toks[ix].startswith('##'):
                if tokenizer.convert_tokens_to_ids(['##' + toks[ix]]) != [tokenizer.unk_token_id]:
                    toks[ix] = '##' + toks[ix]
                else:
                    toks.insert(ix, "##¤")  # add special token to indicate next token is connected 29647 ['##¤']

        raw_tokens += toks

    tokens, labels = [], []
    tkn_pos = 1
    for tkn in raw_tokens:
        if tkn.startswith('##'):
            labels.append('X')
            tokens.append(tkn)
            tkn_pos += 1
        elif tkn in label_list:
            try:
                if labels:
                    labels[-tkn_pos] = tkn
            except Exception as e:
                logger.warning(f'Exception {e.args} caught while processing:\n {text}')

            # TODO: try to change into
            # if labels:
            #     labels[-tkn_pos:] = [tkn] * tkn_pos

        else:
            labels.append('O')
            tokens.append(tkn)
            tkn_pos = 1
    assert len(tokens) == len(labels)
    return tokens, labels


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


class BertForTokenClassification(BertPreTrainedModel):
    """BERT model for token-level classification.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForTokenClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, weight=None):
        super().__init__(config)

        self.bert = BertModel(config)  # TODO: this also inits super, BertModel extends our parent - we should just inherit BertModel or just wrap it without inheritence
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.weight = weight
        self.num_labels = config.num_labels
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=self.weight)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self, unfreeze=['10', '11', 'pooler']):
        self.freeze_bert_encoder()
        for name, param in self.bert.named_parameters():
            if any(t in name for t in unfreeze):
                logger.info(f'Unfreezing {name}')
                param.requires_grad = True


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def train(args, train_dataset, model, device):
    """ Train the model """
    if args.local_rank in [-1, 0] and 'monitor_train' in args:
        tb_writer = SummaryWriter()

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # if args.max_steps > 0:
    #     t_total = args.max_steps
    #     args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    # else:
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=10, t_total=t_total)

    # multi-gpu training (should be after apex fp16 initialization)
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    # logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    # train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    for _ in range(int(args.num_train_epochs)):   # train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':       batch[0],
                      'attention_mask':  batch[1],
                      'token_type_ids':  None if args.model_type == 'xlm' else batch[2],
                      'labels':          batch[3]
                      }
            # if args.model_type in ['xlnet', 'xlm']:
            #     inputs.update({'cls_index': batch[5],
            #                    'p_mask':       batch[6]})
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # if args.fp16:
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            # else:

            loss.backward()  # auto-grad
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                    #     results = evaluate(args, model, tokenizer)
                    #     for key, value in results.items():
                    #         tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    if 'monitor_train' in args:
                        tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        # if args.max_steps > 0 and global_step > args.max_steps:
        #     train_iterator.close()
        #     break

    if args.local_rank in [-1, 0] and 'monitor_train' in args:
        tb_writer.close()

    return global_step, tr_loss / global_step


def load_and_cache_examples(args, tokenizer, evaluate=False):
    # if args.local_rank not in [-1, 0] and not evaluate:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    input_file = args.eval_file if evaluate else args.train_file
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}'.format(
        'eval' if evaluate else 'train',
        args.bert_model_dir.stem,
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        features = convert_file_to_features(input_file, ['.', '?', ','], tokenizer, args.max_seq_length, skip=(not evaluate))

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    dataset = features_to_dataset(features, evaluate)
    return dataset


def features_to_dataset(features, evaluate):
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_lbl_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    if evaluate:
        # all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lbl_ids)
    else:
        # all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        # all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)

        # dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
        #                         all_start_positions, all_end_positions, all_lbl_ids)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lbl_ids)
    return dataset


def predict(args, eval_dataset, model, device, disable_progressbar=False):
    eval_sampler = SequentialSampler(eval_dataset) #if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    true_labels = None
    all_logits = None
    eval_loss, eval_accuracy = 0, 0

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Prediction Iteration", disable=disable_progressbar)):
        input_ids, input_mask, segment_ids, label_ids = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        true_labels = label_ids.numpy().ravel() if true_labels is None else np.concatenate((true_labels, label_ids.numpy().ravel()))

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)[0]
            #logits = logits.sigmoid()  # convert to probabilities

        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

    all_logits = all_logits.reshape((all_logits.shape[0]*all_logits.shape[1], -1))
    all_logits = torch.Tensor(all_logits).softmax(dim=1).numpy()
    predictions = np.argmax(all_logits, axis=1).ravel()
    return true_labels, predictions, all_logits


def _to_feature(tokens, labels, tokenizer, label_map, max_seq_length):
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    labels_ids = [float(label_map[label]) for label in labels]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding
    labels_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_ids=labels_ids)


def convert_file_to_features(fname, label_list, tokenizer, max_seq_length, skip=False):
    """Loads a data file into a list of `InputBatch`s."""
    with open(fname) as fin:
        text = fin.read()
        if hasattr(fname, 'name'):
            fname = fname.name
        if '.timed-text.' in fname:
            text = re.sub(r'\[.+\>', '<EOS>', text)
            text = re.sub(r'\d+:\d+:\d+.\d+\s', '', text)
            text = text.replace('\t', ' ')
            # print(text)
        text = text.replace(' $ ', '')

    segments = text.split('<EOS>')
    return convert_segments_to_features(segments, label_list, tokenizer, max_seq_length, skip)


def convert_segments_to_features(segments, label_list, tokenizer, max_seq_length, skip=False, disable_progressbar=False):
    label_map = {label: i for i, label in enumerate(['O', 'X'] + label_list)}

    features = []
    tokens, labels = ['[CLS]'], ['X']
    seg_idx, skipped = 0, 0

    with tqdm(desc='Text monologues:', total=len(segments), disable=disable_progressbar) as tbar:
        while seg_idx < len(segments):
            tbar.update(1)
            if not segments[seg_idx] or (skip and segments[seg_idx].find('?')<0 and np.random.rand()<=0.5):
                seg_idx += 1
                skipped += 1
                continue
            seg_tokens, seg_labels = tokenize(segments[seg_idx], tokenizer, label_list)
            # if len(seg_tokens) < 2:
            #     seg_idx += 1
            #     continue
            while len(seg_tokens) >= max_seq_length:
                if len(tokens)>1:
                    features.append(_to_feature(tokens, labels, tokenizer, label_map, max_seq_length))
                    tokens, labels = ['[CLS]'], ['X']

                tokens += seg_tokens[:max_seq_length-2] + ['[SEP]']
                labels += seg_labels[:max_seq_length-2] + ['X']
                seg_tokens = seg_tokens[max_seq_length-2:]
                seg_labels = seg_labels[max_seq_length-2:]

            while len(tokens) + len(seg_tokens) >= max_seq_length:
                if len(tokens)>1:
                    features.append(_to_feature(tokens, labels, tokenizer, label_map, max_seq_length))
                    tokens, labels = ['[CLS]'], ['X']

                tokens += seg_tokens[:max_seq_length-2] + ['[SEP]']
                labels += seg_labels[:max_seq_length-2] + ['X']
                seg_tokens = seg_tokens[max_seq_length-2:]
                seg_labels = seg_labels[max_seq_length-2:]

            if seg_tokens:
                tokens += seg_tokens[:max_seq_length-2] + ['[SEP]']
                labels += seg_labels[:max_seq_length-2] + ['X']
            seg_idx += 1

        if len(tokens) > 1:
            features.append(_to_feature(tokens, labels, tokenizer, label_map, max_seq_length))

    if skipped:
        logger.info(f'Skipped {skipped} sentences')
    return features


def normalize(tokenizer, ids, predictions, scores, split_by_monolog=False):
    """ Convert ids back to word (taking care of splitted words) """
    words, klas, skores = [], [], []

    tokens = tokenizer.convert_ids_to_tokens(ids)

    add_linefeed = False
    idx = -1
    while idx < len(tokens)-1:
        idx += 1
        token = tokens[idx]

        if token in ['[CLS]', '[PAD]']:
            continue
        if token == '[SEP]':
            if split_by_monolog and idx<len(tokens)-1:
                add_linefeed = True
            continue

        if token.startswith('##'):
            tkn = words.pop()
            if token=='##¤':
                idx += 1
                token = tokens[idx]
                words.append(tkn + token)
            else:
                words.append(tkn + token[2:])
        else:
            words.append(token)
            if add_linefeed:
                words[-1] = '\n' + words[-1]
            klas.append(predictions[idx])
            if scores is not None:
                skores.append(scores[idx])
            else:
                skores.append(None)

        add_linefeed = False

    assert len(words) == len(klas)
    assert len(words) == len(skores)
    return words, klas, skores


def translate_results(tokenizer, ids, labels, scores):
    words, klas, skores = [], [], []

    p_lst = [' ', '', '. ', '? ', ', ']
    tokens = tokenizer.convert_ids_to_tokens(ids)

    try:
        for idx in range(len(tokens)):
            token = tokens[idx]
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue

            if token.startswith('##'):
                tkn = words.pop()
                words.append(tkn + token[2:])
            else:
                if token=="'" or (token=='s' and tokens[idx-1]=="'"):
                    tkn = words.pop()
                    words.append(tkn + token)
                    continue

                words.append(token)
                klas.append(p_lst[labels[idx]])
                if scores is not None:
                    skores.append(scores[idx])
                else:
                    skores.append(None)
    except Exception as ex:
        print(ex)

    assert len(words) == len(klas)
    assert len(words) == len(skores)
    return words, klas, skores

def textify_punctation(tokenizer, ids, labels, scores=None, split=True):
    words, klas, prob = translate_results(tokenizer, ids, labels, scores)

    buf = []
    for ix in range(len(words)):
        buf.append(words[ix])
        buf.append(klas[ix])
        if split and ix>0 and ix%20==0:
            buf.append('\n')

    return ''.join(buf)


def HTMLify_punctuation(tokenizer, ids, true_labels, pred_label, scores):
    def _build_HTML(html_tokens):
        res = '<!DOCTYPE html> <html> <style> \n.del {background-color: papayawhip; color: red;}\n' \
              '.add {background-color:skyblue;} \n.eq {/* background-color:lightgreen; */} \n' \
              '.neq {background-color: lightgrey;}\n' \
              '</style><head></head><body style="font-family:Helvetica; font-size:larger;">\n' \
              '<a class="del">deleted</a><br><a class="add">added</a><br>' \
              '<a class="eq">equal</a><br><a class="neq">non equal</a><br><br>'
        res += ''.join(html_tokens)
        res += '\n</body></html>'
        return res

    def _to_a(token, klass, title=''):
        if not title:
            return f'<a class="{klass}">{token}</a>'
        return f'<a class="{klass}" title="{title}">{token}</a>'

    buff = [' ']

    p_lst = [' ', '', '. ', '? ', ', ']
    tokens = tokenizer.convert_ids_to_tokens(ids)

    for idx in range(len(tokens)):
        token = tokens[idx]

        # if idx % 20 == 0 and idx>0 and buff[-1]!=' <br>\n':
        #     buff.append(' <br>\n')
        if token in ['[CLS]', '[PAD]']:
            continue
        if token == '[SEP]':
            buff.append(' <br><br>\n')
            continue

        if token.startswith('##'):
            lbl = buff.pop()
            tkn = buff.pop()
            buff.append(tkn + token[2:])
            buff.append(lbl)
        else:
            if token=='i' or \
               tokens[idx-1] == '[SEP]' or \
               (('>. ' in buff[-1] or '>? ' in buff[-1]) and 'class="del"' not in buff[-1]):
                  token = token.capitalize()
            if token == "'":
                buff.pop()
                buff.append(token)
                continue

            buff.append(token)

            rng = list(range(len(p_lst)))
            rng.remove(1)   # drop index for splitted tokens
            ttl = ''
            for indx2 in rng:
                ttl += f'{p_lst[indx2]} : {100*scores[idx, indx2]:.0f}\n'

            if true_labels[idx] == pred_label[idx]:
                if pred_label[idx]>1:
                    buff.append(_to_a(p_lst[pred_label[idx]], 'eq', ttl))
                else:
                    buff.append(p_lst[pred_label[idx]])
            elif true_labels[idx] > 1 and pred_label[idx] <= 1:
                buff.append(_to_a(p_lst[true_labels[idx]], 'del', ttl))
            elif true_labels[idx] <= 1 and pred_label[idx] > 1:
                buff.append(_to_a(p_lst[pred_label[idx]], 'add', ttl))
            else:
                buff.append(_to_a(p_lst[pred_label[idx]], 'neq', f'prev: {p_lst[true_labels[idx]]}\n {ttl}'))

    return _build_HTML(buff)


class BertPunctuator():
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
        n_gpu = torch.cuda.device_count()
        args['n_gpu'] = n_gpu
        self.args = args
        logger.info(f"device: {self.device} \t #gpu: {n_gpu}")

        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir.as_posix(), do_lower_case=True)
        self.model = BertForTokenClassification.from_pretrained(args.output_dir.resolve())
        self.model.to(self.device)

        self.input_shape = (0, args.seq_len)
        self.labels = [' ', '', '.', '?', ',']


    def get_punctuation_results(self, sentences, split_by_monolog=True):
        feats = convert_segments_to_features(sentences, self.labels[2:], self.tokenizer, 512, disable_progressbar=True)
        dataset = features_to_dataset(feats, evaluate=True)
        true_labels, predictions, scores = predict(self.args, dataset, self.model, self.device, disable_progressbar=True)

        ids = dataset.tensors[0].numpy().ravel()
        words, klas, prob = normalize(self.tokenizer, ids, predictions, scores, split_by_monolog=split_by_monolog)

        return words, klas, prob

