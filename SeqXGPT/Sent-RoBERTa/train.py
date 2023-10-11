import os
import sys
import json
import torch
import numpy as np
import warnings
import torch.nn.functional as F
import torch.nn as nn


from tqdm import tqdm, trange
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer

warnings.filterwarnings('ignore')

project_path = os.path.abspath('')
if project_path not in sys.path:
    sys.path.append(project_path)
import backend_model_info
from dataloader import DataManager
from model import RoBERTClassifier


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """"
            embeddings: B*E
            labels:     B
        """
        device = embeddings.device
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        sim_matrix = torch.matmul(embeddings_norm, embeddings_norm.t().contiguous())
        labels = labels.view(labels.size(0), 1)

        mask = torch.eq(labels, labels.t().contiguous()).float().to(device)
        mask = mask.fill_diagonal_(0)

        exp_sim_matrix = torch.exp(sim_matrix / self.temperature)

        mask_sum = torch.sum(mask, dim=-1)
        # exp_sim_sum = torch.sum(exp_sim_matrix * mask, dim=-1)
        # denom = torch.sum(exp_sim_matrix, dim=-1) - exp_sim_matrix.diag() - exp_sim_sum
        denom = torch.sum(exp_sim_matrix, dim=-1) - exp_sim_matrix.diag()
        exp_sim_sum = -torch.log(exp_sim_matrix / denom)
        exp_sim_sum = exp_sim_sum * mask        

        loss = torch.zeros_like(mask_sum)
        non_zero_mask = (mask_sum > 0)
        loss[non_zero_mask] = torch.sum(exp_sim_sum[non_zero_mask], dim=-1) / mask_sum[non_zero_mask]
        loss = torch.mean(loss)

        return loss


class SupervisedTrainer:
    def __init__(self, data, model, loss_criterion='CrossEntropyLoss', train_mode=None):
        self.data = data
        self.model = model
        self.loss_criterion = loss_criterion
        self.train_mode = train_mode
        if loss_criterion == 'ContrastiveLoss':
            self.criterion = ContrastiveLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        self.num_train_epochs = 4
        self.weight_decay = 0.1
        self.lr = 1e-5
        self.warm_up_ratio = 0.1

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self._create_optimizer_and_scheduler()

    def _create_optimizer_and_scheduler(self):
        num_training_steps = len(
            self.data.train_dataloader) * self.num_train_epochs
        no_decay = ["bias", "LayerNorm.weight"]

        if self.train_mode == 'Contrastive_Classifier':
            named_parameters = self.model.classifier.named_parameters()
        else:
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

    def train(self, ckpt_name):
        contrastive_learning = (self.loss_criterion == 'ContrastiveLoss')
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
                    logits = self.model(inputs['input_ids'], inputs['masks'], contrastive_learning)
                    labels = inputs['labels']
                    loss = self.criterion(logits, labels)
                    # print(loss.item())
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    tr_loss += loss.item()
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            print('*' * 120)
            print(f'epoch {epoch+1}: train_loss {loss}')
            # test
            if self.loss_criterion == 'ContrastiveLoss':
                self.test_contrastive()
            else:
                self.test()
            torch.save(self.model.cpu(), ckpt_name)
            self.model.to(self.device)

        torch.save(self.model.cpu(), ckpt_name)
        saved_model = torch.load(ckpt_name)
        self.model = self.model.load_state_dict(saved_model.state_dict())
        return

    def test(self):
        self.model.eval()
        tr_loss = 0
        nb_tr_steps = 0
        true_labels = []
        pred_labels = []
        for step, inputs in enumerate(
                tqdm(self.data.test_dataloader, desc="Iteration")):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
            with torch.no_grad():
                logits = self.model(inputs['input_ids'], inputs['masks'])
                probs = torch.nn.functional.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                labels = inputs['labels']
                loss = self.criterion(logits, labels)
                pred_labels.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())

                tr_loss += loss.item()
                nb_tr_steps += 1

        loss = tr_loss / nb_tr_steps

        p_r_sent_file = ""
        with open(p_r_sent_file, 'w', encoding='utf-8') as f:
            json.dump({"true_labels": true_labels, "pred_labels": pred_labels}, f)
        
        self._get_precision_recall_acc_macrof1(true_labels, pred_labels)
        # true_labels = np.array(true_labels)
        # pred_labels = np.array(pred_labels)
        # acc_label = precision_score(y_true=true_labels,
        #                             y_pred=pred_labels,
        #                             average=None)
        # rec_label = recall_score(y_true=true_labels,
        #                         y_pred=pred_labels,
        #                         average=None)
        # acc = (true_labels == pred_labels).astype(
        #     np.float32).mean().item()

        print(f'test_loss: {loss}')
        # print("Total acc: {}".format(acc))
        # print("The accuracy of each class:")
        # print(acc_label)
        # print("The recall of each class:")
        # print(rec_label)

    def test_contrastive(self):
        contrastive_learning = (self.loss_criterion == 'ContrastiveLoss')
        self.model.eval()
        tr_loss = 0
        nb_tr_steps = 0
        for step, inputs in enumerate(
                tqdm(self.data.test_dataloader, desc="Iteration")):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
            with torch.no_grad():
                logits = self.model(inputs['input_ids'], inputs['masks'], contrastive_learning)
                labels = inputs['labels']
                loss = self.criterion(logits, labels)
                tr_loss += loss.item()
                nb_tr_steps += 1
        loss = tr_loss / nb_tr_steps
        print(f'test_loss: {loss}')

    def _get_precision_recall_acc_macrof1(self, true_labels, pred_labels):
        accuracy = accuracy_score(true_labels, pred_labels)
        macro_f1 = f1_score(true_labels, pred_labels, average='macro')
        print("Accuracy: {:.1f}".format(accuracy*100))
        print("Macro F1 Score: {:.1f}".format(macro_f1*100))

        precision = precision_score(true_labels, pred_labels, average=None)
        recall = recall_score(true_labels, pred_labels, average=None)
        print("Precision/Recall per class: ")
        precision_recall = ' & '.join(["{:.1f}/{:.1f}".format(p*100, r*100) for p, r in zip(precision, recall)])
        print(precision_recall)

        result = {"precision":precision, "recall":recall, "accuracy":accuracy, "macro_f1":macro_f1}
        return result

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        type=str,
        required=False,
        default="roberta-base",
        help="The base-model to use.",
    )
    parser.add_argument(
        "--train-mode",
        type=str,
        required=False,
        default="classify",
        help="",
    )
    parser.add_argument("--gpu",
                        type=str,
                        required=False,
                        default='0',
                        help="Set os.environ['CUDA_VISIBLE_DEVICES'].")
    return parser.parse_args()


# TODO: set the train_path and the test_path
if __name__ == "__main__":
    train_path = ''
    test_path = ''
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    tokenizer = RobertaTokenizer.from_pretrained(args.base_model)
    # data = DataManager(train_path=train_path, test_path=test_path, tokenizer=tokenizer)
    
    """linear classify"""
    if args.train_mode == 'classify':
        print('-' * 32 + 'classify' + '-' * 32)
        # binary classification
        data = DataManager(train_path=train_path, test_path=test_path, tokenizer=tokenizer)

        classifier = RoBERTClassifier(base_model_name=args.base_model, class_num=6)
        ckpt_name = ''
        trainer = SupervisedTrainer(data, classifier)

        print("Log INFO: do test...")
        saved_model = torch.load(ckpt_name)
        trainer.model.load_state_dict(saved_model.state_dict())
        trainer.test()

        # trainer.train(ckpt_name=ckpt_name)

        # classifier = RoBERTClassifier(base_model_name=args.base_model, class_num=backend_model_info.en_class_num)
        # trainer = SupervisedTrainer(data, classifier)
        # trainer.train(ckpt_name=ckpt_name)

    """contrastive training"""
    if args.train_mode == 'contrastive_learning':
        print('-' * 32 + 'contrastive_learning' + '-' * 32)
        pass

    """classify after contrastive"""
    if args.train_mode == 'contrastive_classify':
        print('-' * 32 + 'contrastive_classify' + '-' * 32)
        pass