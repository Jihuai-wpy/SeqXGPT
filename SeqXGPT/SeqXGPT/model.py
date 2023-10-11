import torch
import torch.nn as nn

from typing import List, Tuple
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers.models.bert import BertModel
from fastNLP.modules.torch import MLP,ConditionalRandomField,allowed_transitions
from torch.nn import CrossEntropyLoss


class ConvFeatureExtractionModel(nn.Module):

    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        conv_dropout: float = 0.0,
        conv_bias: bool = False,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride=1, conv_bias=False):
            padding = k // 2
            return nn.Sequential(
                nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=k, stride=stride, padding=padding, bias=conv_bias),
                nn.Dropout(conv_dropout),
                # nn.BatchNorm1d(n_out),
                nn.ReLU(),
                # nn.MaxPool1d(kernel_size=2, stride=2)
            )

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for _, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(in_d, dim, k, stride=stride, conv_bias=conv_bias))
            in_d = dim

    def forward(self, x):
        # x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x


class ModelWiseCNNClassifier(nn.Module):

    def __init__(self, id2labels, dropout_rate=0.1):
        super(ModelWiseCNNClassifier, self).__init__()
        feature_enc_layers = [(64, 5, 1)] + [(128, 3, 1)] * 3 + [(64, 3, 1)]
        self.conv = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            conv_dropout=0.0,
            conv_bias=False,
        )

        embedding_size = 4 *64
        self.norm = nn.LayerNorm(embedding_size)
        
        self.label_num = len(id2labels)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(nn.Linear(embedding_size, self.label_num))
        self.crf = ConditionalRandomField(num_tags=self.label_num, allowed_transitions=allowed_transitions(id2labels))
        self.crf.trans_m.data *= 0

    def conv_feat_extract(self, x):
        out = self.conv(x)
        out = out.transpose(1, 2)
        return out

    def forward(self, x, labels):
        x = x.transpose(1, 2)
        out1 = self.conv_feat_extract(x[:, 0:1, :])  
        out2 = self.conv_feat_extract(x[:, 1:2, :])  
        out3 = self.conv_feat_extract(x[:, 2:3, :])  
        out4 = self.conv_feat_extract(x[:, 3:4, :])  
        outputs = torch.cat((out1, out2, out3, out4), dim=2)  
        
        outputs = self.norm(outputs)
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
    

class ModelWiseTransformerClassifier(nn.Module):

    def __init__(self, id2labels, seq_len, intermediate_size = 512, num_layers=2, dropout_rate=0.1):
        super(ModelWiseTransformerClassifier, self).__init__()
        # feature_enc_layers = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]
        feature_enc_layers = [(64, 5, 1)] + [(128, 3, 1)] * 3 + [(64, 3, 1)]
        self.conv = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            conv_dropout=0.0,
            conv_bias=False,
        )
        
        self.seq_len = seq_len          # MAX Seq_len
        embedding_size = 4 *64
        self.encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=16,
            dim_feedforward=intermediate_size,
            dropout=dropout_rate,
            batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer=self.encoder_layer,
                                            num_layers=num_layers)

        self.position_encoding = torch.zeros((seq_len, embedding_size))
        for pos in range(seq_len):
            for i in range(0, embedding_size, 2):
                self.position_encoding[pos, i] = torch.sin(
                    torch.tensor(pos / (10000**((2 * i) / embedding_size))))
                self.position_encoding[pos, i + 1] = torch.cos(
                    torch.tensor(pos / (10000**((2 *
                                                 (i + 1)) / embedding_size))))
        
        self.norm = nn.LayerNorm(embedding_size)
        
        self.label_num = len(id2labels)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(nn.Linear(embedding_size, self.label_num))
        self.crf = ConditionalRandomField(num_tags=self.label_num, allowed_transitions=allowed_transitions(id2labels))
        self.crf.trans_m.data *= 0

    def conv_feat_extract(self, x):
        out = self.conv(x)
        out = out.transpose(1, 2)
        return out

    def forward(self, x, labels):
        mask = labels.gt(-1)
        padding_mask = ~mask

        x = x.transpose(1, 2)
        out1 = self.conv_feat_extract(x[:, 0:1, :])  
        out2 = self.conv_feat_extract(x[:, 1:2, :])  
        out3 = self.conv_feat_extract(x[:, 2:3, :])  
        out4 = self.conv_feat_extract(x[:, 3:4, :])  
        out = torch.cat((out1, out2, out3, out4), dim=2)  
        
        outputs = out + self.position_encoding.to(out.device)
        outputs = self.norm(outputs)
        outputs = self.encoder(outputs, src_key_padding_mask=padding_mask)
        dropout_outputs = self.dropout(outputs)
        logits = self.classifier(dropout_outputs)
        
        if self.training:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.label_num), labels.view(-1))
            output = {'loss': loss, 'logits': logits}
        else:
            paths, scores = self.crf.viterbi_decode(logits=logits, mask=mask)
            paths[mask==0] = -1
            output = {'preds': paths, 'logits': logits}
            pass

        return output
    

class TransformerOnlyClassifier(nn.Module):

    def __init__(self, id2labels, seq_len, embedding_size=4, num_heads=2, intermediate_size=64, num_layers=2, dropout_rate=0.1):
        super(TransformerOnlyClassifier, self).__init__()

        self.encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout_rate,
            batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer=self.encoder_layer,
                                            num_layers=num_layers)

        self.position_encoding = torch.zeros((seq_len, embedding_size))
        for pos in range(seq_len):
            for i in range(0, embedding_size, 2):
                self.position_encoding[pos, i] = torch.sin(
                    torch.tensor(pos / (10000**((2 * i) / embedding_size))))
                self.position_encoding[pos, i + 1] = torch.cos(
                    torch.tensor(pos / (10000**((2 *
                                                 (i + 1)) / embedding_size))))
        
        self.norm = nn.LayerNorm(embedding_size)
        
        self.label_num = len(id2labels)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(nn.Linear(embedding_size, self.label_num))
        self.crf = ConditionalRandomField(num_tags=self.label_num, allowed_transitions=allowed_transitions(id2labels))
        self.crf.trans_m.data *= 0
    
    def forward(self, inputs, labels):
        mask = labels.gt(-1)
        padding_mask = ~mask
        
        outputs = inputs + self.position_encoding.to(inputs.device)
        outputs = self.norm(outputs)
        outputs = self.encoder(outputs, src_key_padding_mask=padding_mask)
        dropout_outputs = self.dropout(outputs)
        logits = self.classifier(dropout_outputs)
        
        if self.training:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.label_num), labels.view(-1))
            output = {'loss': loss, 'logits': logits}
        else:
            paths, scores = self.crf.viterbi_decode(logits=logits, mask=mask)
            paths[mask==0] = -1
            output = {'preds': paths, 'logits': logits}
            pass

        return output