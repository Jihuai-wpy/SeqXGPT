import torch
import torch.nn as nn

from transformers import AutoModel
from torch.nn import CrossEntropyLoss
from fastNLP.modules.torch import MLP,ConditionalRandomField,allowed_transitions

class RoBERTClassifier(nn.Module):
    def __init__(self, base_model_name, id2label, dropout_rate=0.1):
        super(RoBERTClassifier, self).__init__()
        self.roberta = AutoModel.from_pretrained(base_model_name)
        self.hidden_size = self.roberta.config.hidden_size
        self.label_num = len(id2label)

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(nn.Linear(self.hidden_size, self.label_num))
        self.crf = ConditionalRandomField(num_tags=self.label_num, allowed_transitions=allowed_transitions(id2label))
        self.crf.trans_m.data *= 0

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        outputs = outputs.last_hidden_state
        
        dropout_outputs = self.dropout(outputs)
        logits = self.classifier(dropout_outputs)
        
        if self.training:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.label_num), labels.view(-1))
            output = {'loss': loss, 'logits': logits}
        else:
            mask = labels.gt(-1)
            paths, scores = self.crf.viterbi_decode(logits=logits, mask=mask)
            paths[mask==0] = -1
            output = {'preds': paths, 'logits': logits}
            pass

        return output