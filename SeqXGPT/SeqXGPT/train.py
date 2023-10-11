import os
import sys
import json
import torch
import numpy as np
import warnings
import torch.nn.functional as F
import torch.nn as nn


from tqdm import tqdm, trange
from sklearn.metrics import precision_score, recall_score
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

warnings.filterwarnings('ignore')

project_path = os.path.abspath('')
if project_path not in sys.path:
    sys.path.append(project_path)
import backend_model_info
from dataloader import DataManager
from model import ModelWiseCNNClassifier, ModelWiseTransformerClassifier, TransformerOnlyClassifier



class SupervisedTrainer:
    def __init__(self, data, model, en_labels, id2label, args):
        self.data = data
        self.model = model
        self.en_labels = en_labels
        self.id2label =id2label

        self.seq_len = args.seq_len
        self.num_train_epochs = args.num_train_epochs
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.warm_up_ratio = args.warm_up_ratio

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.model.to(self.device)
        self._create_optimizer_and_scheduler()

    def _create_optimizer_and_scheduler(self):
        num_training_steps = len(
            self.data.train_dataloader) * self.num_train_epochs
        no_decay = ["bias", "LayerNorm.weight"]

        named_parameters = self.model.named_parameters()
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in named_parameters
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                self.weight_decay,
            },
            {
                "params": [
                    p for n, p in named_parameters
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            betas=(0.9, 0.98),
            eps=1e-8,
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warm_up_ratio * num_training_steps,
            num_training_steps=num_training_steps)

    def train(self, ckpt_name='linear_en.pt'):
        for epoch in trange(int(self.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_steps = 0
            # train
            for step, inputs in enumerate(
                    tqdm(self.data.train_dataloader, desc="Iteration")):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                with torch.set_grad_enabled(True):
                    labels = inputs['labels']
                    output = self.model(inputs['features'], inputs['labels'])
                    logits = output['logits']
                    loss = output['loss']
                    # print(loss.item())
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    tr_loss += loss.item()
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            print(f'epoch {epoch+1}: train_loss {loss}')
            # test
            self.test()
            print('*' * 120)
            torch.save(self.model.cpu(), ckpt_name)
            self.model.to(self.device)

        torch.save(self.model.cpu(), ckpt_name)
        saved_model = torch.load(ckpt_name)
        self.model.load_state_dict(saved_model.state_dict())
        return

    def test(self, content_level_eval=False):
        self.model.eval()
        texts = []
        true_labels = []
        pred_labels = []
        total_logits = []
        for step, inputs in enumerate(
                tqdm(self.data.test_dataloader, desc="Iteration")):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
            with torch.no_grad():
                labels = inputs['labels']
                output = self.model(inputs['features'], inputs['labels'])
                logits = output['logits']
                preds = output['preds']
                
                texts.extend(inputs['text'])
                pred_labels.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
                total_logits.extend(logits.cpu().tolist())
        
        # with open("", 'w') as f:
        #     f.write(json.dumps(total_logits[3], ensure_ascii=False) + '\n')
        #     f.write(json.dumps(texts[3], ensure_ascii=False) + '\n')
        #     f.write(json.dumps(true_labels[3], ensure_ascii=False) + '\n')
        #     f.write(json.dumps(pred_labels[3], ensure_ascii=False) + '\n')


        if content_level_eval:
            # content level evaluation
            print("*" * 8, "Content Level Evalation", "*" * 8)
            content_result = self.content_level_eval(texts, true_labels, pred_labels)
        else:
            # sent level evalation
            print("*" * 8, "Sentence Level Evalation", "*" * 8)
            sent_result = self.sent_level_eval(texts, true_labels, pred_labels)

        # word level evalation
        print("*" * 8, "Word Level Evalation", "*" * 8)
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)
        true_labels_1d = true_labels.reshape(-1)
        pred_labels_1d = pred_labels.reshape(-1)
        mask = true_labels_1d != -1
        true_labels_1d = true_labels_1d[mask]
        pred_labels_1d = pred_labels_1d[mask]
        accuracy = (true_labels_1d == pred_labels_1d).astype(np.float32).mean().item()
        print("Accuracy: {:.1f}".format(accuracy*100))
        pass
    
    def content_level_eval(self, texts, true_labels, pred_labels):
        from collections import Counter

        true_content_labels = []
        pred_content_labels = []
        for text, true_label, pred_label in zip(texts, true_labels, pred_labels):
            true_label = np.array(true_label)
            pred_label = np.array(pred_label)
            mask = true_label != -1
            true_label = true_label[mask].tolist()
            pred_label = pred_label[mask].tolist()
            true_common_tag = self._get_most_common_tag(true_label)
            true_content_labels.append(true_common_tag[0])
            pred_common_tag = self._get_most_common_tag(pred_label)
            pred_content_labels.append(pred_common_tag[0])
        
        true_content_labels = [self.en_labels[label] for label in true_content_labels]
        pred_content_labels = [self.en_labels[label] for label in pred_content_labels]
        result = self._get_precision_recall_acc_macrof1(true_content_labels, pred_content_labels)
        return result

    def sent_level_eval(self, texts, true_labels, pred_labels):
        """
        """
        true_sent_labels = []
        pred_sent_labels = []
        for text, true_label, pred_label in zip(texts, true_labels, pred_labels):
            true_sent_label = self.get_sent_label(text, true_label)
            pred_sent_label = self.get_sent_label(text, pred_label)
            true_sent_labels.extend(true_sent_label)
            pred_sent_labels.extend(pred_sent_label)
        
        true_sent_labels = [self.en_labels[label] for label in true_sent_labels]
        pred_sent_labels = [self.en_labels[label] for label in pred_sent_labels]
        result = self._get_precision_recall_acc_macrof1(true_sent_labels, pred_sent_labels)
        return result

    def get_sent_label(self, text, label):
        import nltk
        sent_separator = nltk.data.load('tokenizers/punkt/english.pickle')
        sents = sent_separator.tokenize(text)

        offset = 0
        sent_label = []
        for sent in sents:
            start = text[offset:].find(sent) + offset
            end = start + len(sent)
            offset = end
            
            split_sentence = self.data.split_sentence
            end_word_idx = len(split_sentence(text[:end]))
            if end_word_idx > self.seq_len:
                break
            word_num = len(split_sentence(text[start:end]))
            start_word_idx = end_word_idx - word_num
            tags = label[start_word_idx:end_word_idx]
            most_common_tag = self._get_most_common_tag(tags)
            sent_label.append(most_common_tag[0])
        
        if len(sent_label) == 0:
            print("empty sent label list")
        return sent_label
    
    def _get_most_common_tag(self, tags):
        """most_common_tag is a tuple: (tag, times)"""
        from collections import Counter

        tags = [self.id2label[tag] for tag in tags]
        tags = [tag.split('-')[-1] for tag in tags]
        tag_counts = Counter(tags)
        most_common_tag = tag_counts.most_common(1)[0]

        return most_common_tag

    def _get_precision_recall_acc_macrof1(self, true_labels, pred_labels):
        accuracy = accuracy_score(true_labels, pred_labels)
        macro_f1 = f1_score(true_labels, pred_labels, average='macro')
        print("Accuracy: {:.1f}".format(accuracy*100))
        print("Macro F1 Score: {:.1f}".format(macro_f1*100))

        precision = precision_score(true_labels, pred_labels, average=None)
        recall = recall_score(true_labels, pred_labels, average=None)
        print("Precision/Recall per class: ")
        precision_recall = ' '.join(["{:.1f}/{:.1f}".format(p*100, r*100) for p, r in zip(precision, recall)])
        print(precision_recall)

        result = {"precision":precision, "recall":recall, "accuracy":accuracy, "macro_f1":macro_f1}
        return result


def construct_bmes_labels(labels):
    prefix = ['B-', 'M-', 'E-', 'S-']
    id2label = {}
    counter = 0

    for label, id in labels.items():
        for pre in prefix:
            id2label[counter] = pre + label
            counter += 1
    
    return id2label

def split_dataset(data_path, train_path, test_path, train_ratio=0.9):
    file_names = [file_name for file_name in os.listdir(data_path) if file_name.endswith('.jsonl')]
    print('*'*32)
    print('The overall data sources:')
    print(file_names)
    file_paths = [os.path.join(data_path, file_name) for file_name in file_names]

    total_samples = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            samples = [json.loads(line) for line in f]
            total_samples.extend(samples)
    
    import random
    random.seed(0)
    random.shuffle(total_samples)

    split_index = int(len(total_samples) * train_ratio)
    train_data = total_samples[:split_index]
    test_data = total_samples[split_index:]

    def save_dataset(fpath, data_samples):
        with open(fpath, 'w', encoding='utf-8') as f:
            for sample in tqdm(data_samples):
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    save_dataset(train_path, train_data)
    save_dataset(test_path, test_data)
    print()
    print("The number of train dataset:", len(train_data))
    print("The number of test  dataset:", len(test_data))
    print('*'*32)
    pass

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Transformer')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--train_mode', type=str, default='classify')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=1024)

    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--split_dataset', action='store_true')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--train_path', type=str, default='')
    parser.add_argument('--test_path', type=str, default='')

    parser.add_argument('--num_train_epochs', type=int, default=20)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warm_up_ratio', type=float, default=0.1)

    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--test_content', action='store_true')
    return parser.parse_args()

# python ./Seq_train/train.py --gpu=0 --split_dataset
# python ./Seq_train/train.py --gpu=0
if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.split_dataset:
        print("Log INFO: split dataset...")
        split_dataset(data_path=args.data_path, train_path=args.train_path, test_path=args.test_path, train_ratio=args.train_ratio)

    # en_labels = backend_model_info.en_labels
    en_labels = {
        'gpt2': 0,
        'gptneo': 1,
        'gptj': 2,
        'llama': 3,
        'gpt3re': 4,
        # 'gpt3sum': 3,
        'human': 5
    }
    # en_labels = {'AI':0, 'human':1}

    id2label = construct_bmes_labels(en_labels)
    label2id = {v: k for k, v in id2label.items()}

    data = DataManager(train_path=args.train_path, test_path=args.test_path, batch_size=args.batch_size, max_len=args.seq_len, human_label='human', id2label=id2label)
    
    """linear classify"""
    if args.train_mode == 'classify':
        print('-' * 32 + 'classify' + '-' * 32)
        if args.model == 'CNN':
            print('-' * 32 + "CNN" + '-' * 32)
            classifier = ModelWiseCNNClassifier(id2labels=id2label)
            ckpt_name = ''
        elif args.model == 'RNN':
            print('-' * 32 + "RNN" + '-' * 32)
            classifier = TransformerOnlyClassifier(id2labels=id2label, seq_len=args.seq_len)
            ckpt_name = ''
        else:
            classifier = ModelWiseTransformerClassifier(id2labels=id2label, seq_len=args.seq_len)
            ckpt_name = ''

        trainer = SupervisedTrainer(data, classifier, en_labels, id2label, args)

        if args.do_test:    
            print("Log INFO: do test...")
            saved_model = torch.load(ckpt_name)
            trainer.model.load_state_dict(saved_model.state_dict())
            trainer.test(content_level_eval=args.test_content)
        else:
            print("Log INFO: do train...")
            trainer.train(ckpt_name=ckpt_name)

    """contrastive training"""
    if args.train_mode == 'contrastive_learning':
        print('-' * 32 + 'contrastive_learning' + '-' * 32)
        if args.model == 'CNN':
            classifier = ModelWiseCNNClassifier(class_num=backend_model_info.en_class_num)
            ckpt_name = ''
        else:
            classifier = ModelWiseTransformerClassifier(class_num=backend_model_info.en_class_num)
            ckpt_name = ''

        trainer = SupervisedTrainer(data, classifier, loss_criterion = 'ContrastiveLoss')
        trainer.train(ckpt_name=ckpt_name)

    """classify after contrastive"""
    if args.train_mode == 'contrastive_classify':
        print('-' * 32 + 'contrastive_classify' + '-' * 32)
        if args.model == 'CNN':
            classifier = ModelWiseCNNClassifier(class_num=backend_model_info.en_class_num)
            ckpt_name = ''
            saved_model = torch.load(ckpt_name)
            classifier.load_state_dict(saved_model.state_dict())
            ckpt_name = ''
        else:
            classifier = ModelWiseTransformerClassifier(class_num=backend_model_info.en_class_num)
            ckpt_name = ''
            saved_model = torch.load(ckpt_name)
            classifier.load_state_dict(saved_model.state_dict())
            ckpt_name = ''

        # trainer = SupervisedTrainer(data, classifier, train_mode='Contrastive_Classifier')
        trainer = SupervisedTrainer(data, classifier)
        trainer.train(ckpt_name=ckpt_name)