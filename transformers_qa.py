""" QA finetuning on hugginface.transformers """
import argparse
import os
import random
import logging
import re
from time import time
from logging.config import dictConfig
from typing import List
from itertools import chain
from pprint import pprint

import transformers
import torch
import numpy as np
from torch import nn
from torch.autograd import detect_anomaly
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from get_dataset import Dataset, get_dataset_qa
from checkpoint_versioning import Argument
from compute_logit_qa import compute_logits, compute_gold_answer, get_evaluation


dictConfig({
    "version": 1,
    "formatters": {'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}},
    "handlers": {'h': {'class': 'logging.StreamHandler', 'formatter': 'f', 'level': logging.DEBUG}},
    "root": {'handlers': ['h'], 'level': logging.DEBUG}})
LOGGER = logging.getLogger()
NUM_WORKER = int(os.getenv("NUM_WORKER", '4'))
PROGRESS_INTERVAL = int(os.getenv("PROGRESS_INTERVAL", '50'))
CACHE_DIR = os.getenv("CACHE_DIR", './cache')
PAD_TOKEN_LABEL_ID = nn.CrossEntropyLoss().ignore_index
SAMPLE_N = None  # sample size for validation while run training
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning
os.makedirs(CACHE_DIR, exist_ok=True)


class Transforms:
    """ QA specific transform pipeline """
    models_need_p_mask = ["xlnet", "xlm"]
    models_need_lang = ["xlm"]

    def __init__(self, transformer_tokenizer: str):
        """ QA specific transform pipeline """
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_tokenizer, cache_dir=CACHE_DIR)
        self.name = type(self.tokenizer).__name__.lower().replace('tokenizer', '')
        # num of special token for prefix/suffix/sent-pair
        self.n_sp_token_pre = self.__n_sp_token_prefix()
        self.n_sp_token_suf = self.tokenizer.max_len - self.tokenizer.max_len_single_sentence - self.n_sp_token_pre
        self.n_sp_token_pair = self.tokenizer.max_len - self.tokenizer.max_len_sentences_pair - (
                self.n_sp_token_pre + self.n_sp_token_suf)
        LOGGER.info('setup {0} tokenizer (max_len: {1})'.format(transformer_tokenizer, self.tokenizer.max_len))

    def __n_sp_token_prefix(self):
        sentence_go_around = ''.join(self.tokenizer.tokenize('get tokenizer specific prefix'))
        return len(sentence_go_around[:list(re.finditer('get', sentence_go_around))[0].span()[0]])

    @staticmethod
    def is_max_context(doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span["start"] + doc_span["length"] - 1
            if position < doc_span["start"] or position > end:
                continue
            num_left_context = position - doc_span["start"]
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index
        return cur_span_index == best_span_index

    def get_p_mask(self, input_ids, query_encode, length, cls_index):
        """ p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        Original TF implem also keep the classification token (set to 0)
        eg) XLNet, XLM """
        p_mask = np.ones_like(input_ids)
        if self.tokenizer.padding_side == "right":
            p_mask[len(query_encode) + self.n_sp_token_pre + self.n_sp_token_pair:] = 0
        else:
            p_mask[-len(length): -(len(query_encode) + self.n_sp_token_pre + self.n_sp_token_pair)] = 0
        pad_token_indices = np.where(input_ids == self.tokenizer.pad_token_id)
        special_token_indices = np.asarray(
            self.tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
        ).nonzero()
        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1
        # Set the cls index to 0: the CLS index can be used for impossible answers
        p_mask[cls_index] = 0
        return p_mask

    def combine_query_context(self, query, context, max_context_len, doc_stride):
        """ Define the side we want to truncate / pad and the text/pair sorting """
        if len(context) > max_context_len:
            context_remain = context[max_context_len - doc_stride:]
            context = context[:max_context_len]
        else:
            context_remain = []
        if self.tokenizer.padding_side == "right":  # bert family
            texts = query
            pairs = context
            doc_offset = len(query) + self.n_sp_token_pair + self.n_sp_token_pre
        else:
            texts = context
            pairs = query
            doc_offset = 0

        return texts, pairs, doc_offset, len(context), context_remain

    def encode_plus_all(self,
                        context: List,
                        question: List,
                        question_id: List=None,
                        doc_stride: int = 128,
                        max_length: int = 384,
                        max_query_length: int = 64,
                        answer: List = None,
                        with_negative: bool = False):
        """ encode all the list """
        max_length = self.tokenizer.max_len if max_length is None else max_length
        shared_param = {'doc_stride': doc_stride, 'max_query_length': max_query_length, 'max_length': max_length,
                        'pad_to_max_length': True, 'with_negative': with_negative}
        question_id = [str(i) for i in range(len(context))] if question_id is None else question_id
        encode_info = [self.encode_plus(
            context=context[i],
            question=question[i],
            question_id=question_id[i],
            answer=None if answer is None else answer[i],
            **shared_param) for i in range(len(context))]
        encode = list(chain(*[e for e, i in encode_info]))
        info = list(chain(*[i for e, i in encode_info]))
        return encode, info

    def encode_plus(self,
                    context: str,
                    question: str,
                    question_id: str,
                    answer: List = None,
                    doc_stride: int = 128,
                    max_length: int = 384,
                    max_query_length: int = 64,
                    pad_to_max_length: bool = True,
                    with_negative: bool = False):
        """ return (encode_dicts, info_dicts) where first one is for model input, and the other has additional info """
        assert doc_stride < max_length

        query_encode = self.tokenizer.encode(
            question, add_special_tokens=False, truncation=True, max_length=max_query_length)

        # `is_impossible` is global flag, where `_is_impossible` is when answer can't be found in overflow tokens
        if answer is None or len(answer) == 0:
            is_impossible = None if answer is None else True
            start_position = end_position = None
            context_encode = self.tokenizer.encode(context, add_special_tokens=False)
        else:
            is_impossible = False
            start, end = answer
            forward_tokens = self.tokenizer.encode(context[:start], add_special_tokens=False)
            mention_tokens = self.tokenizer.encode(context[start:end], add_special_tokens=False)
            backward_tokens = self.tokenizer.encode(context[end:], add_special_tokens=False)
            context_encode = forward_tokens + mention_tokens + backward_tokens
            start_position = len(forward_tokens)
            end_position = start_position + len(mention_tokens)

        encoded_dict_chunks = []  # list of encodes that can be fed into model directly
        info_dict_chunks = []  # list of info, required when compute logit
        n_stride = -1
        max_context_len = max_length - len(query_encode) - (
                self.n_sp_token_pair + self.n_sp_token_pre + self.n_sp_token_suf)
        while True:
            if len(context_encode) == 0:
                break
            n_stride += 1
            # `tokenizers` 's truncation and overflowing pipeline have consistent errors, so not to use it
            texts, pairs, doc_offset, context_length, context_encode = self.combine_query_context(
                query_encode, context_encode, max_context_len, doc_stride)
            encoded_dict = self.tokenizer.encode_plus(
                texts, pairs, pad_to_max_length=pad_to_max_length, max_length=max_length)
            info_dict = {
                "token_is_max_context": {},
                "start": (max_context_len - doc_stride) * n_stride,  # stride start point
                "question_id": question_id,
                "length": context_length,
                "doc_offset": doc_offset
            }

            if is_impossible is None:
                # inference mode, without answer
                info_dict['is_impossible'] = is_impossible
            else:
                # Identify the position of the CLS token
                cls_index = encoded_dict["input_ids"].index(self.tokenizer.cls_token_id)

                # is_impossible answers are designed to predict CLS
                encoded_dict["start_positions"] = cls_index
                encoded_dict["end_positions"] = cls_index
                _is_impossible = True  # impossible flag for overflow tokens
                if not is_impossible:
                    fixed_start_position = start_position + doc_offset - info_dict['start']
                    fixed_end_position = end_position + doc_offset - info_dict['start']
                    _is_impossible = True
                    if fixed_start_position >= doc_offset and fixed_end_position <= max_length - self.n_sp_token_suf:
                        encoded_dict["start_positions"] = fixed_start_position
                        encoded_dict["end_positions"] = fixed_end_position
                        _is_impossible = False
                if _is_impossible and not with_negative:
                    continue

                info_dict['is_impossible'] = _is_impossible
                if self.name in self.models_need_p_mask:
                    encoded_dict["cls_index"] = cls_index
                    encoded_dict["p_mask"] = self.get_p_mask(
                        encoded_dict["input_ids"], query_encode, context_length, encoded_dict["cls_index"])
                    encoded_dict["is_impossible"] = _is_impossible

            if self.name in self.models_need_lang:
                encoded_dict["langs"] = [0] * len(encoded_dict["input_ids"])  # 0 for english
            encoded_dict_chunks.append(encoded_dict)
            info_dict_chunks.append(info_dict)

        for n, info_dict in enumerate(info_dict_chunks):
            for j in range(info_dict["length"]):
                is_max_context = self.is_max_context(info_dict_chunks, n, n * doc_stride + j)
                if self.tokenizer.padding_side == 'right':
                    index = len(query_encode) + self.n_sp_token_pre + self.n_sp_token_pair + j
                else:
                    index = j
                info_dict["token_is_max_context"][index] = is_max_context
        return encoded_dict_chunks, info_dict_chunks


class TrainTransformerQA:
    """ finetune transformers on QA """

    def __init__(self,
                 batch_size_validation: int = None,
                 checkpoint: str = None,
                 checkpoint_dir: str = './ckpt',
                 **kwargs):
        LOGGER.info('*** initialize network ***')

        # checkpoint version
        self.args = Argument(checkpoint=checkpoint, checkpoint_dir=checkpoint_dir, **kwargs)
        self.batch_size_validation = batch_size_validation if batch_size_validation else self.args.batch_size

        # fix random seed
        random.seed(self.args.random_seed)
        transformers.set_seed(self.args.random_seed)
        torch.manual_seed(self.args.random_seed)
        torch.cuda.manual_seed_all(self.args.random_seed)

        # dataset
        self.dataset_split = get_dataset_qa(
            self.args.dataset, cache_dir=CACHE_DIR, v2_with_negative=self.args.with_negative)

        # model setup
        self.model = transformers.AutoModelForQuestionAnswering.from_pretrained(
            self.args.transformer,
            config=transformers.AutoConfig.from_pretrained(self.args.transformer, cache_dir=CACHE_DIR))
        self.transforms = Transforms(self.args.transformer)

        # optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}]
        self.optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=1e-8)

        # scheduler
        self.scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.args.warmup_step, num_training_steps=self.args.total_step)

        # apply checkpoint statistics to optimizer/scheduler
        self.__step = 0
        self.__epoch = 0
        self.__best_val_score = None
        if self.args.model_statistics is not None:
            self.__step = self.args.model_statistics['step']
            self.__epoch = self.args.model_statistics['epoch']
            self.__best_val_score = self.args.model_statistics['best_val_score']
            self.model.load_state_dict(self.args.model_statistics['model_state_dict'])
            if self.optimizer is not None and self.scheduler is not None:
                self.optimizer.load_state_dict(self.args.model_statistics['optimizer_state_dict'])
                self.scheduler.load_state_dict(self.args.model_statistics['scheduler_state_dict'])

        # GPU allocation
        self.n_gpu = torch.cuda.device_count()
        self.device = 'cuda' if self.n_gpu > 0 else 'cpu'
        self.model.to(self.device)

        # GPU mixture precision
        self.scale_loss, self.master_params = None, None
        if self.args.fp16:
            try:
                from apex import amp  # noqa: F401
                self.model, self.optimizer = amp.initialize(
                    self.model, self.optimizer, opt_level='O1', max_loss_scale=2**13, min_loss_scale=1e-5)
                self.master_params = amp.master_params
                self.scale_loss = amp.scale_loss
                LOGGER.info('using `apex.amp`')
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        # multi-gpus
        if self.n_gpu > 1:
            # multi-gpu training (should be after apex fp16 initialization)
            self.model = torch.nn.DataParallel(self.model.cuda())
            LOGGER.info('using `torch.nn.DataParallel`')
        LOGGER.info('running on %i GPUs' % self.n_gpu)

    def __setup_loader(self, data_type: str):
        assert data_type in self.dataset_split.keys()
        is_train = data_type == 'train'
        encode, info_list = self.transforms.encode_plus_all(
            context=self.dataset_split[data_type]['context'],
            question=self.dataset_split[data_type]['question'],
            question_id=self.dataset_split[data_type]['question_id'],
            answer=self.dataset_split[data_type]['answer'],
            doc_stride=self.args.doc_stride,
            max_query_length=self.args.max_query_length,
            max_length=self.args.max_seq_length if is_train else None,
            with_negative=self.args.with_negative)
        data_obj = Dataset(encode)
        _batch_size = self.args.batch_size if is_train else self.batch_size_validation
        data_loader = torch.utils.data.DataLoader(
            data_obj, num_workers=NUM_WORKER, batch_size=_batch_size, shuffle=is_train, drop_last=is_train)
        info_loader = {k: [i[k] for i in info_list] for k in info_list[0].keys()}
        return data_loader, info_loader

    def test(self):
        LOGGER.addHandler(logging.FileHandler(os.path.join(self.args.checkpoint_dir, 'logger_test.log')))
        loader = {k: self.__setup_loader(k) for k in self.dataset_split.keys() if k != 'train'}
        LOGGER.info('data_loader: %s' % str(list(loader.keys())))
        start_time = time()
        for k, (data_loader, info_loader) in loader.items():
            self.__epoch_valid(data_loader, info_loader=info_loader, prefix=k)
            self.release_cache()
        LOGGER.info('[test completed, %0.2f sec in total]' % (time() - start_time))

    def train(self):
        LOGGER.addHandler(logging.FileHandler(os.path.join(self.args.checkpoint_dir, 'logger_train.log')))
        writer = SummaryWriter(log_dir=self.args.checkpoint_dir)
        start_time = time()

        # setup dataset/data loader
        loader = {k: self.__setup_loader(k) for k in ['train', 'valid']}
        LOGGER.info('data_loader: %s' % str(list(loader.keys())))

        # start training
        LOGGER.info('*** start training from step %i, epoch %i ***' % (self.__step, self.__epoch))
        try:
            with detect_anomaly():
                while True:

                    data_loader, _ = loader['train']
                    if_training_finish = self.__epoch_train(
                        data_loader, writer=writer)
                    self.release_cache()

                    data_loader, info_loader = loader['valid']
                    self.__epoch_valid(
                        data_loader, info_loader=info_loader, writer=writer, prefix='valid', sample_n=SAMPLE_N)
                    self.release_cache()

                    if if_training_finish:
                        break

                    self.__epoch += 1
        except RuntimeError:
            LOGGER.exception('*** RuntimeError (NaN found, see above log in detail) ***')

        except KeyboardInterrupt:
            LOGGER.info('*** KeyboardInterrupt ***')

        self.__save()
        LOGGER.info('[training completed, %0.2f sec in total]' % (time() - start_time))
        writer.close()
        LOGGER.info('ckpt saved at %s' % self.args.checkpoint_dir)

    def __save(self):
        if self.n_gpu > 1:
            model_wts = self.model.module.state_dict()
        else:
            model_wts = self.model.state_dict()
        torch.save({
            'best_val_score': self.__best_val_score,
            'step': self.__step,
            'epoch': self.__epoch,
            'model_state_dict': model_wts,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, os.path.join(self.args.checkpoint_dir, 'model.pt'))

    def __epoch_train(self, data_loader, writer):
        """ train on single epoch, returning flag which is True if training has been completed """
        self.model.train()
        for i, encode in enumerate(data_loader, 1):
            # update model
            self.optimizer.zero_grad()
            loss = self.model(**{k: v.to(self.device) for k, v in encode.items()})[0]
            if self.n_gpu > 1:
                loss = loss.mean()
            if self.args.fp16:
                with self.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.master_params(self.optimizer), self.args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            # optimizer and scheduler step
            self.optimizer.step()
            self.scheduler.step()
            # log instantaneous accuracy, loss, and learning rate
            inst_loss = loss.cpu().detach().item()
            inst_lr = self.optimizer.param_groups[0]['lr']
            writer.add_scalar('train/loss', inst_loss, self.__step)
            writer.add_scalar('train/learning_rate', inst_lr, self.__step)
            if self.__step % PROGRESS_INTERVAL == 0:
                LOGGER.info('[epoch %i] * (training step %i) loss: %.3f, lr: %0.8f'
                            % (self.__epoch, self.__step, inst_loss, inst_lr))
            self.__step += 1
            # break
            if self.__step >= self.args.total_step:
                LOGGER.info('reached maximum step')
                return True
        return False

    def __epoch_valid(self, data_loader, info_loader, sample_n: int = None, writer=None, prefix: str='valid'):
        """ validation/test, returning flag which is True if early stop condition was applied """
        self.model.eval()
        start_golds = []
        end_golds = []
        start_logits = []
        end_logits = []
        input_ids = []

        LOGGER.info('process {} dataset...'.format(prefix))
        for encode in tqdm(data_loader):
            input_ids += encode['input_ids'].cpu().detach().int().tolist()
            model_outputs = self.model(**{k: v.to(self.device) for k, v in encode.items()})
            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the others only use two.
            if len(model_outputs) >= 5:
                # loss, start_logits, start_top_index, end_logits, end_top_index, cls_logits = model_outputs[:6]
                raise ValueError('not implemented yet')
            _, _start_logits, _end_logits = model_outputs[:3]
            start_logits += _start_logits.cpu().detach().int().tolist()
            end_logits += _end_logits.cpu().detach().int().tolist()
            start_golds += encode['start_positions'].detach().int().tolist()
            end_golds += encode['end_positions'].detach().int().tolist()
            if sample_n is not None and len(input_ids) >= sample_n:
                LOGGER.info('use {0}/{1} entries only'.format(len(input_ids), len(data_loader.dataset)))
                break

        # get prediction
        LOGGER.info('compute logit...')
        all_nbest = compute_logits(
            start_logits=start_logits,
            end_logits=end_logits,
            question_id=info_loader['question_id'][:len(input_ids)],
            length=info_loader['length'][:len(input_ids)],
            doc_offset=info_loader['doc_offset'][:len(input_ids)],
            token_is_max_context=info_loader['token_is_max_context'][:len(input_ids)],
            input_ids=input_ids,
            tokenizer=self.transforms.tokenizer)

        # correct answer
        LOGGER.info('aggregate correct...')
        all_gold = compute_gold_answer(
            start_positions=start_golds,
            end_positions=end_golds,
            question_id=info_loader['question_id'][:len(input_ids)],
            input_ids=input_ids,
            tokenizer=self.transforms.tokenizer)

        # get metric
        metric = get_evaluation(all_gold, all_nbest)
        LOGGER.info('[epoch {0}] ({1}):'.format(self.__epoch, prefix))
        for k, v in metric.items():
            LOGGER.info(' - {0}: {1}'.format(k, v))
            if writer:
                writer.add_scalar('{0}/{1}'.format(k, v), self.__epoch)
        return False

    def release_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()


class TransformerQA:
    """ transformers QA, interface to get prediction from pre-trained checkpoint """

    def __init__(self, checkpoint: str):
        LOGGER.info('*** initialize network ***')

        # checkpoint version
        self.args = Argument(checkpoint=checkpoint)
        self.model = transformers.AutoModelForQuestionAnswering.from_pretrained(
            self.args.transformer,
            config=transformers.AutoConfig.from_pretrained(self.args.transformer, cache_dir=CACHE_DIR))
        self.transforms = Transforms(self.args.transformer)
        self.model.load_state_dict(self.args.model_statistics['model_state_dict'])

        # GPU allocation
        self.n_gpu = torch.cuda.device_count()
        self.device = 'cuda' if self.n_gpu > 0 else 'cpu'
        self.model.to(self.device)
        LOGGER.info('running on %i GPUs' % self.n_gpu)

    def predict(self,
                context: List,
                question: List,
                doc_stride: int = 128,
                max_length: int = 384,
                max_query_length: int = 64):
        self.model.eval()
        assert len(context) == len(question), 'context size should be same as question size'
        encode_list, info_list = self.transforms.encode_plus_all(
            context=context,
            question=question,
            doc_stride=doc_stride,
            max_length=max_length,
            max_query_length=max_query_length)
        data_loader = torch.utils.data.DataLoader(Dataset(encode_list), batch_size=len(encode_list))
        info_loader = {k: [i[k] for i in info_list] for k in info_list[0].keys()}

        start_logits = []
        end_logits = []
        input_ids = []
        for encode in tqdm(data_loader):
            input_ids += encode['input_ids'].cpu().detach().int().tolist()
            model_outputs = self.model(**{k: v.to(self.device) for k, v in encode.items()})
            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the others only use two.
            if len(model_outputs) >= 5:
                # loss, start_logits, start_top_index, end_logits, end_top_index, cls_logits = model_outputs[:6]
                raise ValueError('not implemented yet')
            _start_logits, _end_logits = model_outputs[:2]
            start_logits += _start_logits.cpu().detach().int().tolist()
            end_logits += _end_logits.cpu().detach().int().tolist()

        # get prediction
        LOGGER.info('compute logit...')
        all_nbest = compute_logits(
            start_logits=start_logits,
            end_logits=end_logits,
            question_id=info_loader['question_id'][:len(input_ids)],
            length=info_loader['length'][:len(input_ids)],
            doc_offset=info_loader['doc_offset'][:len(input_ids)],
            token_is_max_context=info_loader['token_is_max_context'][:len(input_ids)],
            input_ids=input_ids,
            tokenizer=self.transforms.tokenizer)
        answer = [all_nbest[i] for i in sorted(all_nbest.keys())]
        return answer


def get_options():
    parser = argparse.ArgumentParser(
        description='finetune transformers to sentiment analysis',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', '--checkpoint', help='checkpoint to load', default=None, type=str)
    parser.add_argument('--checkpoint-dir', help='checkpoint directory', default='./ckpt', type=str)
    parser.add_argument('-d', '--data', help='squad-v1 or squad-v2', default='squad-v1', type=str)
    parser.add_argument('-t', '--transformer', help='pretrained language model', default='xlm-roberta-base', type=str)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_seq_length", default=384, type=int, help="The maximum total input sequence length.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question.")
    parser.add_argument('-b', '--batch-size', help='batch size', default=8, type=int)
    parser.add_argument('--lr', help='learning rate', default=3e-5, type=float)
    parser.add_argument('--random-seed', help='random seed', default=1234, type=int)
    parser.add_argument('--total-step', help='total training step', default=22000, type=int)
    parser.add_argument('--batch-size-validation',
                        help='batch size for validation (smaller size to save memory)',
                        default=2,
                        type=int)
    parser.add_argument('--warmup-step', help='warmup step (6% of total is recommended)', default=100, type=int)
    parser.add_argument('--weight-decay', help='weight decay', default=0.0, type=float)
    parser.add_argument('--test', help='run over testdataset', action='store_true')
    parser.add_argument('--fp16', help='fp16', action='store_true')
    parser.add_argument('--with-negative', help='use negative samples', action='store_true')
    parser.add_argument('--inference-mode', help='inference mode', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()
    if not opt.inference_mode:
        # train model
        trainer = TrainTransformerQA(
            max_seq_length=opt.max_seq_length,
            max_query_length=opt.max_query_length,
            doc_stride=opt.doc_stride,
            batch_size_validation=opt.batch_size_validation,
            checkpoint=opt.checkpoint,
            checkpoint_dir=opt.checkpoint_dir,
            dataset=opt.data,
            transformer=opt.transformer,
            random_seed=opt.random_seed,
            lr=opt.lr,
            total_step=opt.total_step,
            warmup_step=opt.warmup_step,
            weight_decay=opt.weight_decay,
            batch_size=opt.batch_size,
            fp16=opt.fp16,
            max_grad_norm=opt.max_grad_norm,
            with_negative=opt.with_negative
        )
        if opt.test:
            trainer.test()
        else:
            trainer.train()
    else:
        # play around with trained model
        classifier = TransformerQA(checkpoint=opt.checkpoint)
        test_context = ["""
        Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL)
        for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National
        Football Conference (NFC) champion Carolina Panthers 2410 to earn their third Super Bowl title. The game was
        played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this
        was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives,
        as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which
        the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic
        numerals 50.
        """]
        test_question = ["Which NFL team represented the AFC at Super Bowl 50?"]
        test_result = classifier.predict(context=test_context, question=test_question)
        pprint('-- DEMO --')
        pprint(test_result)
        pprint('----------')
        while True:
            _context = input('input context >>>')
            _question = input('input question >>>')
            if _context == 'q' or _question == 'q':
                break
            elif _context == '' or _question == '':
                continue
            else:
                pprint(classifier.predict(context=[_context], question=[_question]))


