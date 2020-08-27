""" Fetch dataset for NLP task """
import os
import logging
import json
from logging.config import dictConfig
from typing import List

import torch
from tqdm import tqdm


dictConfig({
    "version": 1,
    "formatters": {'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}},
    "handlers": {'h': {'class': 'logging.StreamHandler', 'formatter': 'f', 'level': logging.DEBUG}},
    "root": {'handlers': ['h'], 'level': logging.DEBUG}})
LOGGER = logging.getLogger()
STOPWORDS = ['None', '#']
__all__ = ("Dataset", "get_dataset_qa")


class Dataset(torch.utils.data.Dataset):
    """ simple torch.utils.data.Dataset wrapper converting into tensor"""
    float_tensors = ['attention_mask']

    def __init__(self, data: List):
        self.data = data

    def __len__(self):
        return len(self.data)

    def to_tensor(self, name, data):
        if name in self.float_tensors:
            return torch.tensor(data, dtype=torch.float32)
        return torch.tensor(data, dtype=torch.long)

    def __getitem__(self, idx):
        return {k: self.to_tensor(k, v) for k, v in self.data[idx].items()}


def get_dataset_qa(data_name: str = 'squad-v1', cache_dir: str = './cache', v2_with_negative: bool = True):
    def get_squad(data_path, v2: bool = False):
        os.makedirs(data_path, exist_ok=True)
        if v2:
            url_train = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json'
            url_valid = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json'
        else:
            url_train = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json'
            url_valid = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json'
        if not os.path.exists(os.path.join(data_path, 'train.json')):
            os.system('wget -O {0} {1}'.format(os.path.join(data_path, 'train.json'), url_train))
        if not os.path.exists(os.path.join(data_path, 'dev.json')):
            os.system('wget -O {0} {1}'.format(os.path.join(data_path, 'dev.json'), url_valid))

        def _prepare_squad(_file: str):
            data = json.load(open(_file, 'r'))['data']
            context, question, answer, question_id = [], [], [], []
            for i in tqdm(data):
                for p in i['paragraphs']:
                    for q in p['qas']:
                        if v2 and q['is_impossible'] and v2_with_negative:
                            # is_impossible.append(True)
                            context.append(p['context'])
                            question.append(q['question'])
                            answer.append([])
                            question_id.append(q['id'])
                        else:
                            # remove duplication from a list of dict, since answers are duplicated in validation set
                            for a in [dict(t) for t in {tuple(d.items()) for d in q['answers']}]:
                                # is_impossible.append(False)
                                context.append(p['context'])
                                question.append(q['question'])
                                answer.append([a['answer_start'], a['answer_start'] + len(a['text'])])
                                question_id.append(q['id'])
            LOGGER.info('file: {0}'.format(_file))
            LOGGER.info(' * total: {0} (negative: {1})'.format(len(context), sum(len(i) == 0 for i in answer)))
            return {'context': context, 'question': question, 'question_id': question_id, 'answer': answer}

        valid = _prepare_squad(os.path.join(data_path, 'dev.json'))
        train = _prepare_squad(os.path.join(data_path, 'train.json'))
        return {'train': train, 'valid': valid}

    if data_name in ['squad-v1', 'squad-v2']:
        return get_squad(os.path.join(cache_dir, data_name), v2=data_name == 'squad-v2')
    else:
        raise ValueError('unknown dataset: {}'.format(data_name))

