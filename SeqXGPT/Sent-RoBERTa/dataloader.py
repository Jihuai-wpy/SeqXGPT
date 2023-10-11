import numpy as np
import os
import re
import random
import torch
import json
import pickle
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from torch.utils.data.dataloader import DataLoader, RandomSampler, SequentialSampler
from sklearn.preprocessing import normalize


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


batch_size = 32

class DataManager:

    def __init__(self, train_path, test_path, tokenizer, max_len=512):
        set_seed(0)
        self.tokenizer = tokenizer
        self.max_len = max_len
        data = dict()

        if train_path:
            train_dict = self.initialize_dataset(train_path)
            data["train"] = Dataset.from_dict(train_dict)
        
        if test_path:
            test_dict = self.initialize_dataset(test_path)
            data["test"] = Dataset.from_dict(test_dict)
        
        
        datasets = DatasetDict(data)
        if train_path:
            self.train_dataloader = self.get_train_dataloader(datasets["train"])
        if test_path:
            self.test_dataloader = self.get_eval_dataloader(datasets["test"])

    def get_train_dataloader(self, dataset):
        return DataLoader(dataset,
                          batch_size=batch_size,
                          sampler=RandomSampler(dataset),
                          collate_fn=self.data_collator)

    def get_eval_dataloader(self, dataset):
        return DataLoader(dataset,
                          batch_size=batch_size,
                          sampler=SequentialSampler(dataset),
                          collate_fn=self.data_collator)
    
    def initialize_dataset(self, data_path, save_dir=''):
        processed_data_filename = Path(data_path).stem + "_processed.pkl"
        processed_data_path = os.path.join(save_dir, processed_data_filename)

        # if os.path.exists(processed_data_path):
        #     print('*' * len('*'*4 + 'Load From {}'.format(processed_data_path) + '*'*4))
        #     print('*'*4, 'Load From {}'.format(processed_data_path), '*'*4)
        #     print('*' * len('*'*4 + 'Load From {}'.format(processed_data_path) + '*'*4))
        #     with open(processed_data_path, 'rb') as f:
        #         samples_dict = pickle.load(f)
        #     return samples_dict

        with open(data_path, 'r') as f:
            if data_path.endswith('json'):
                samples = json.load(f)
            else:
                samples = [json.loads(line) for line in f]

        samples_dict = {'input_ids': [], 'labels': [], 'text': []}
        for item in tqdm(samples):
            text = item['text']
            label = item['label']
            # label_int = item['label_int']

            # solve the multiclass classification problem
            en_labels = {
                'gpt2': 0,
                'gptneo': 1,
                'gptj': 2,
                'llama': 3,
                'gpt3re': 4,
                'gpt3sum': 4,
                'human': 5
            }
            label_int = en_labels[label]

            input_ids = self.tokenizer(text).input_ids

            samples_dict['input_ids'].append(input_ids)
            samples_dict['labels'].append(label_int)
            samples_dict['text'].append(text)
        
        # with open(processed_data_path, 'wb') as f:
        #     pickle.dump(samples_dict, f)

        return samples_dict
    
    def data_collator(self, samples):
        pad_token_id = self.tokenizer.pad_token_id
        batch = {}

        tokenized_ids = [sample['input_ids'] for sample in samples]

        max_len = max([len(ids) for ids in tokenized_ids])
        max_len = min(max_len, self.max_len)

        input_ids = np.ones((len(tokenized_ids), max_len), dtype=int) * pad_token_id
        masks = np.zeros((len(tokenized_ids), max_len), dtype=int)

        for idx, ids in enumerate(tokenized_ids):
            length = min(len(ids), max_len)
            input_ids[idx, :length] = ids[:length]
            masks[idx, :length] = 1

        batch['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
        batch['masks'] = torch.tensor(masks, dtype=torch.long)

        labels = [sample['labels'] for sample in samples]
        batch["labels"] = torch.tensor(labels)

        batch['text'] = [sample['text'] for sample in samples]

        return batch