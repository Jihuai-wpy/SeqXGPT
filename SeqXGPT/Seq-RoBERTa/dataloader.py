import numpy as np
import os
import random
import torch
import json
import pandas as pd
import pickle

from tqdm import tqdm
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from torch.utils.data.dataloader import DataLoader, RandomSampler, SequentialSampler
from sklearn.preprocessing import normalize


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class DataManager:

    def __init__(self, train_path, test_path, batch_size, max_len, human_label, id2label, tokenizer, label_pad_idx=-1, load_from_cache=False):
        set_seed(0)
        self.batch_size = batch_size
        self.max_len = max_len
        self.human_label = human_label
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.tokenizer = tokenizer
        self.label_pad_idx = label_pad_idx

        data = dict()

        if train_path:
            # {'features': [], 'prompt_len': [], 'label_int': [], 'text': []}
            train_dict = self.initialize_dataset(train_path, load_from_cache)
            data["train"] = Dataset.from_dict(train_dict)
        
        if test_path:
            test_dict = self.initialize_dataset(test_path, load_from_cache)
            data["test"] = Dataset.from_dict(test_dict)
        
        datasets = DatasetDict(data)
        if train_path:
            self.train_dataloader = self.get_train_dataloader(datasets["train"])
        else:
            self.train_dataloader = None
        if test_path:
            self.test_dataloader = self.get_eval_dataloader(datasets["test"])
        else:
            self.test_dataloader = None

    def initialize_dataset(self, data_path, load_from_cache, save_dir=''):
        processed_data_filename = Path(data_path).stem + "_roberta_processed.pkl"
        processed_data_path = os.path.join(save_dir, processed_data_filename)

        if os.path.exists(processed_data_path) and load_from_cache:
            log_info = '*'*4 + 'Load From {}'.format(processed_data_path) + '*'*4
            print('*' * len(log_info))
            print(log_info)
            print('*' * len(log_info))
            with open(processed_data_path, 'rb') as f:
                samples_dict = pickle.load(f)
            return samples_dict

        with open(data_path, 'r') as f:
            if data_path.endswith('json'):
                samples = json.load(f)
            else:
                samples = [json.loads(line) for line in f]

        samples_dict = {'input_ids': [], 'masks': [], 'labels': [], 'is_not_last_sent':[], 'text': []}
        tokenizer = self.tokenizer

        for item in tqdm(samples):
            text = item['text']
            prompt_len = item['prompt_len']
            label = item['label']

            if label == "gpt3sum":
                label = 'gpt3re'
            # prompt_len = 0

            label_int = item['label_int']

            words = self.split_sentence(text)

            total_len = len(words)
            prefix_len = len(self.split_sentence(text[:prompt_len]))
            prefix_labels = self.sequence_labels_to_ids(prefix_len, self.human_label)
            suffix_labels = self.sequence_labels_to_ids(total_len - prefix_len, label)
            labels = prefix_labels + suffix_labels

            aligned_ids = []
            aligned_labels = []
            for word, label in zip(words, labels):
                sub_tokens = tokenizer.tokenize(word)
                aligned_ids.extend(tokenizer.convert_tokens_to_ids(sub_tokens))
                aligned_labels.extend([label] + [self.label_pad_idx] * (len(sub_tokens)-1))
            assert len(aligned_ids) == len(aligned_labels), "len(aligned_ids) != len(aligned_labels), something error."

            def split_list(lst, max_len):
                return [lst[i:i+max_len] for i in range(0, len(lst), max_len)]
            
            input_ids_list = split_list(aligned_ids, self.max_len)
            labels_list = split_list(aligned_labels, self.max_len)

            masks_list = [[1] * self.max_len for _ in range(len(input_ids_list))]

            last_list_len = len(input_ids_list[-1])
            input_ids_list[-1] += [tokenizer.pad_token_id] * (self.max_len - last_list_len)
            labels_list[-1] += [self.label_pad_idx] * (self.max_len - last_list_len)
            masks_list[-1] = [1] * last_list_len + [0] * (self.max_len - last_list_len)
            is_not_last_sent_list = [1] * (len(input_ids_list) - 1) + [0]
            texts = [text for _ in range(len(input_ids_list))]

            samples_dict['input_ids'].extend(input_ids_list)
            samples_dict['masks'].extend(masks_list)
            samples_dict['labels'].extend(labels_list)
            samples_dict['is_not_last_sent'].extend(is_not_last_sent_list)
            samples_dict['text'].extend(texts)
        
        dir_name = os.path.dirname(processed_data_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            
        with open(processed_data_path, 'wb') as f:
            pickle.dump(samples_dict, f)

        return samples_dict

    def sequence_labels_to_ids(self, seq_len, label):
        prefix = ['B-', 'M-', 'E-', 'S-']
        if seq_len <= 0:
            return []
        elif seq_len == 1:
            label = 'S-' + label
            return [self.label2id[label]]
        else:
            ids = []
            ids.append(self.label2id['B-'+label])
            ids.extend([self.label2id['M-'+label]] * (seq_len - 2))
            ids.append(self.label2id['E-'+label])
            return ids

    def get_train_dataloader(self, dataset):
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          sampler=RandomSampler(dataset),
                          collate_fn=self.data_collator)

    def get_eval_dataloader(self, dataset):
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          sampler=SequentialSampler(dataset),
                          collate_fn=self.data_collator)
    
    def data_collator(self, samples):
        # samples: {'input_ids': [], 'masks': [], 'labels': [], 'is_not_last_sent':[], 'text': []}
        batch = {}

        input_ids = [sample['input_ids'] for sample in samples]
        masks = [sample['masks'] for sample in samples]
        labels = [sample['labels'] for sample in samples]
        is_not_last_sent = [sample['is_not_last_sent'] for sample in samples]
        text = [sample['text'] for sample in samples]

        batch['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
        batch['masks'] = torch.tensor(masks, dtype=torch.long)
        batch['labels'] = torch.tensor(labels, dtype=torch.long)
        batch['is_not_last_sent'] = torch.tensor(is_not_last_sent, dtype=torch.long)
        batch['text'] = text

        return batch


    def _split_en_sentence(self, sentence, use_sp=False):
        import re
        pattern = re.compile(r'\S+|\s')
        words = pattern.findall(sentence)
        if use_sp:
            words = ["▁" if item == " " else item for item in words]
        return words


    def _split_cn_sentence(self, sentence, use_sp=False):
        words = list(sentence)
        if use_sp:
            words = ["▁" if item == " " else item for item in words]
        return words


    def split_sentence(self, sentence, use_sp=False, cn_percent=0.2):
        total_char_count = len(sentence)
        total_char_count += 1 if total_char_count == 0 else 0
        chinese_char_count = sum('\u4e00' <= char <= '\u9fff' for char in sentence)
        if chinese_char_count / total_char_count > cn_percent:
            return self._split_cn_sentence(sentence, use_sp)
        else:
            return self._split_en_sentence(sentence, use_sp)
