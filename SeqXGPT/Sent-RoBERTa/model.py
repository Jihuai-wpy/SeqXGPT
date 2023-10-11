import torch
import torch.nn as nn
from transformers import AutoModel

class RoBERTClassifier(nn.Module):
    def __init__(self, base_model_name, class_num=5, dropout_rate=0.1):
        super(RoBERTClassifier, self).__init__()
        self.roberta = AutoModel.from_pretrained(base_model_name)
        self.hidden_size = self.roberta.config.hidden_size

        self.contrastive_dense = nn.Sequential( nn.Linear(self.hidden_size, self.hidden_size * 4),
                                                nn.Dropout(dropout_rate), nn.ReLU(),
                                                nn.Linear(self.hidden_size * 4, self.hidden_size // 4))
        
        self.classifier = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                        nn.Dropout(dropout_rate), nn.Tanh(),
                                        nn.Linear(self.hidden_size, class_num))

    def forward(self, input_ids, attention_mask=None, contrastive_learning=False):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        if contrastive_learning:
            logits = self.contrastive_dense(cls_embeddings)
            return logits
        
        logits = self.classifier(cls_embeddings)
        return logits
