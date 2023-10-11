import json
import random
import httpx
import msgpack
import numpy as np
import openai
import time
import torch

from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm

import os
import sys

project_path = os.path.abspath('')
if project_path not in sys.path:
    sys.path.append(project_path)
import backend_model_info


def train(samples_train,
          samples_test,
          model_num=4,
          feat_num=4,
          class_num=5,
          ckpt_name='linear_en.pt',
          train_feat='all',
          output_test_set=False):
    # model_num `loss`s  and  C(model_num, 2) `feature`s
    hid_dim = model_num + int(model_num * (model_num - 1) / 2) * feat_num
    # hid_dim = model_num
    # hid_dim = int(model_num * (model_num - 1) / 2)
    # hid_dim = model_num + int(model_num * (model_num - 1) / 2)
    linear_model = torch.nn.Sequential(torch.nn.Linear(hid_dim, 300),
                                       torch.nn.Dropout(0.5), torch.nn.ReLU(),
                                       torch.nn.Linear(300, 300),
                                       torch.nn.Dropout(0.5), torch.nn.ReLU(),
                                       torch.nn.Linear(300, class_num))
    linear_model.to('cuda')
    # use all losses and features
    if train_feat == 'all':
        inputs_train = [x[0] for x in samples_train]
        inputs_test = [x[0] for x in samples_test]

    outputs_train = [x[1] for x in samples_train]
    outputs_test = [x[1] for x in samples_test]

    end_docs = [x[-1] for x in samples_test]

    inputs_train = torch.tensor(inputs_train).to('cuda')
    outputs_train = torch.tensor(outputs_train).to('cuda')

    inputs_test = torch.tensor(inputs_test).to('cuda')
    outputs_test = torch.tensor(outputs_test).to('cuda')

    # train the linear model
    training(linear_model, inputs_train, outputs_train, inputs_test, outputs_test, class_num, end_docs)

    # save the model and load at detector-backend
    torch.save(linear_model.cpu(), ckpt_name)

    # testing
    saved_model = torch.load(ckpt_name)
    linear_model.load_state_dict(saved_model.state_dict())


def training(model, X_train, y_train, X_test, y_test, class_num, end_docs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    n_epochs = 10000

    num = [sum(y_test == i) for i in range(class_num)]
    print(num)

    for it in tqdm(range(n_epochs)):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (it + 1) % 100 == 0:
            with torch.no_grad():
                outputs = model(X_test)
                loss_test = criterion(outputs, y_test)
                prob = torch.nn.functional.softmax(outputs, dim=-1)  # N, label
                pred_labels = torch.argmax(prob, dim=-1)

                true_labels = y_test.cpu().tolist()
                pred_labels = pred_labels.cpu().tolist()

                true_content_labels, pred_content_labels = [], []
                idx = 0
                while idx < len(end_docs):
                    start = idx
                    end = idx + 1
                    while end < len(end_docs) and end_docs[end] == 0:
                        end += 1
                    t_labels = true_labels[start : end]
                    true_common_tag = _get_most_common_tag(t_labels)
                    true_content_labels.append(true_common_tag[0])

                    p_labels = pred_labels[start : end]
                    pred_common_tag = _get_most_common_tag(p_labels)
                    pred_content_labels.append(pred_common_tag[0])
                    idx = end

                print('*' * 120)
                print(
                    f'In this epoch {it+1}/{n_epochs}, Training loss: {loss.item():.4f}, Test loss: {loss_test.item():.4f}'
                )
                _get_precision_recall_acc_macrof1(true_content_labels, pred_content_labels)
    return

def _get_most_common_tag(tags):
        """most_common_tag is a tuple: (tag, times)"""
        from collections import Counter
        tag_counts = Counter(tags)
        most_common_tag = tag_counts.most_common(1)[0]
        return most_common_tag

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
def _get_precision_recall_acc_macrof1(true_labels, pred_labels):
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

def consturct_train_features(samples_train):
    convert_train = []
    en_labels = {
        'gpt2': 0,
        'gptneo': 1,
        'gptj': 2,
        'llama': 3,
        'gpt3re': 4,
        # 'gpt3sum': 4,
        'human': 5
    }
    for item in samples_train:
        label = item['label']
        label_int = en_labels[label]
        end_doc = item.get('end_doc', 1)

        # if label == 'human':
        #     label_int = 1
        # else:
        #     label_int = 0

        values = item['values']
        features = values['losses'] + values['lt_zero_percents'] + values[
            'std_deviations'] + values['pearson_list'] + values[
                'spearmann_list']
        if np.isnan(np.sum(features)):
            continue
        convert_train.append([features, label_int, label, end_doc])
    return convert_train


if __name__ == "__main__":
    name = 'train'

    train_path = ""
    test_path = ""

    with open(train_path, 'r') as f:
        samples_train = [json.loads(line) for line in f]
    with open(test_path, 'r') as f:
        samples_test = [json.loads(line) for line in f]

    # [values, label_int, label]
    samples_train = consturct_train_features(samples_train)
    samples_test = consturct_train_features(samples_test)

    if name == 'train':
        train(
            samples_train=samples_train,
            samples_test=samples_test,
            model_num=4,
            feat_num=4,
            class_num=6,
            ckpt_name=
            '',
            train_feat='all')