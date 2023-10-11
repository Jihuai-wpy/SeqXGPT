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

    def __init__(self, train_path, test_path, batch_size, max_len, human_label, id2label, word_pad_idx=0, label_pad_idx=-1):
        set_seed(0)
        self.batch_size = batch_size
        self.max_len = max_len
        self.human_label = human_label
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx

        data = dict()

        if train_path:
            # {'features': [], 'prompt_len': [], 'label_int': [], 'text': []}
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

    def initialize_dataset(self, data_path, save_dir=''):
        processed_data_filename = Path(data_path).stem + "_processed.pkl"
        processed_data_path = os.path.join(save_dir, processed_data_filename)

        # if os.path.exists(processed_data_path):
        #     log_info = '*'*4 + 'Load From {}'.format(processed_data_path) + '*'*4
        #     print('*' * len(log_info))
        #     print(log_info)
        #     print('*' * len(log_info))
        #     with open(processed_data_path, 'rb') as f:
        #         samples_dict = pickle.load(f)
        #     return samples_dict

        with open(data_path, 'r') as f:
            if data_path.endswith('json'):
                samples = json.load(f)
            else:
                samples = [json.loads(line) for line in f]

        samples_dict = {'features': [], 'prompt_len': [], 'label': [], 'text': []}

        for item in tqdm(samples):
            text = item['text']
            label = item['label']
            prompt_len = item['prompt_len']
            # prompt_len = 0

            # if label in ['gptj', 'gpt2', 'llama', 'gpt3re']:
            #     continue
            # if label == 'gpt3sum':
            #     label = 'gpt3re'
            # if label == 'gpt3re':
            #     continue

            label_int = item['label_int']
            begin_idx_list = item['begin_idx_list']
            ll_tokens_list = item['ll_tokens_list']

            begin_idx_list = np.array(begin_idx_list)
            # Get the maximum value in begin_idx_list, which indicates where we need to truncate.
            max_begin_idx = np.max(begin_idx_list)
            # Truncate all vectors
            for idx, ll_tokens in enumerate(ll_tokens_list):
                ll_tokens_list[idx] = ll_tokens[max_begin_idx:]
            # Get the length of all vectors and take the minimum
            min_len = np.min([len(ll_tokens) for ll_tokens in ll_tokens_list])
            # Align the lengths of all vectors
            for idx, ll_tokens in enumerate(ll_tokens_list):
                ll_tokens_list[idx] = ll_tokens[:min_len]
            if len(ll_tokens_list) == 0 or len(ll_tokens_list[0]) == 0:
                continue
            ll_tokens_list = np.array(ll_tokens_list)
            # ll_tokens_list = normalize(ll_tokens_list, norm='l1')
            ll_tokens_list = ll_tokens_list.transpose()
            ll_tokens_list = ll_tokens_list.tolist()

            samples_dict['features'].append(ll_tokens_list)
            samples_dict['prompt_len'].append(prompt_len)
            samples_dict['label'].append(label)
            samples_dict['text'].append(text)
        
        # with open(processed_data_path, 'wb') as f:
        #     pickle.dump(samples_dict, f)

        return samples_dict


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
        # samples: {'features': [], 'prompt_len': [], 'label': [], 'text': []}
        # batch: {'features': [], 'labels': [], 'text': []}
        batch = {}

        features = [sample['features'] for sample in samples]
        prompt_len = [sample['prompt_len'] for sample in samples]
        text = [sample['text'] for sample in samples]
        label = [sample['label'] for sample in samples]

        features, masks = self.process_and_convert_to_tensor(features)
        # pad_masks = ~masks * -1
        pad_masks = (1 - masks) * self.label_pad_idx

        for idx, p_len in enumerate(prompt_len):
            prefix_len = len(self.split_sentence(text[idx][:p_len]))
            if prefix_len > self.max_len:
                prefix_ids = self.sequence_labels_to_ids(self.max_len, self.human_label)
                masks[idx][:] = prefix_ids[:]
                continue
            total_len = len(self.split_sentence(text[idx]))
            
            if prefix_len > 0:
                prefix_ids = self.sequence_labels_to_ids(prefix_len, self.human_label)
                masks[idx][:prefix_len] = prefix_ids[:]
            if total_len - prefix_len > 0:
                if total_len > self.max_len:
                    human_ids = self.sequence_labels_to_ids(self.max_len - prefix_len, label[idx])
                else:
                    human_ids = self.sequence_labels_to_ids(total_len - prefix_len, label[idx])
                masks[idx][prefix_len:total_len] = human_ids[:]
            masks[idx] += pad_masks[idx]

        batch['features'] = features
        batch['labels'] = masks
        batch['text'] = text

        return batch

    
    def sequence_labels_to_ids(self, seq_len, label):
        prefix = ['B-', 'M-', 'E-', 'S-']
        if seq_len <= 0:
            return None
        elif seq_len == 1:
            label = 'S-' + label
            return torch.tensor([self.label2id[label]], dtype=torch.long)
        else:
            ids = []
            ids.append(self.label2id['B-'+label])
            ids.extend([self.label2id['M-'+label]] * (seq_len - 2))
            ids.append(self.label2id['E-'+label])
            return torch.tensor(ids, dtype=torch.long)

    def process_and_convert_to_tensor(self, data):
        """ here, data is features. """
        max_len = self.max_len
        # data shape: [B, S, E]
        feat_dim = len(data[0][0])
        padded_data = [  # [[0] * feat_dim] + 
            seq + [[0] * feat_dim] * (max_len - len(seq)) for seq in data
        ]
        padded_data = [seq[:max_len] for seq in padded_data]

        # masks = [[False] * min(len(seq)+1, max_len) + [True] * (max_len - min(len(seq)+1, max_len)) for seq in data]
        masks = [[1] * min(len(seq), max_len) + [0] *
                (max_len - min(len(seq), max_len)) for seq in data]

        tensor_data = torch.tensor(padded_data, dtype=torch.float)
        tensor_mask = torch.tensor(masks, dtype=torch.long)

        return tensor_data, tensor_mask


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
