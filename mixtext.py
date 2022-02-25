import torch
import torch.nn as nn
from pytorch_transformers import *
from transformers.modeling_bert import BertEmbeddings, BertPooler, BertLayer


class BertModel4Mix(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel4Mix, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder4Mix(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids,  attention_mask=None, token_type_ids=None,input_ids_1=None,attention_mask_1=None, token_type_ids_1=None, l=None, mix_layer=1000, position_ids=None):

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        if input_ids_1 is not None:
            embedding_output2 = self.embeddings(
                input_ids_1, position_ids=position_ids, token_type_ids=token_type_ids_1)

        if input_ids_1 is not None:
            encoder_outputs = self.encoder(embedding_output, embedding_output2, l, mix_layer,
                                           attention_mask, attention_mask_1)
        else:
            encoder_outputs = self.encoder(
                embedding_output, attention_mask=attention_mask)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs


class BertEncoder4Mix(nn.Module):
    def __init__(self, config):
        super(BertEncoder4Mix, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, hidden_states2=None, l=None, mix_layer=1000, attention_mask=None, attention_mask2=None):
        all_hidden_states = ()
        all_attentions = ()
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        if attention_mask2 !=None:
            attention_mask2 = attention_mask2.unsqueeze(1).unsqueeze(2)
        # Perform mix at till the mix_layer
        if mix_layer == -1:
            if hidden_states2 is not None:
                hidden_states = l * hidden_states + (1-l)*hidden_states2

        for i, layer_module in enumerate(self.layer):
            if i <= mix_layer:

                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                #print(hidden_states.shape, attention_mask.shape)
                layer_outputs = layer_module(
                    hidden_states, attention_mask)
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

                if hidden_states2 is not None:
                    layer_outputs2 = layer_module(
                        hidden_states2, attention_mask2)
                    hidden_states2 = layer_outputs2[0]

            if i == mix_layer:
                if hidden_states2 is not None:
                    hidden_states = l * hidden_states + (1-l)*hidden_states2

            if i > mix_layer:
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states, attention_mask)
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # last-layer hidden state, (all hidden states), (all attentions)
        return outputs


class MixText(nn.Module):
    def __init__(self, num_labels=2, mix_option=False):
        super(MixText, self).__init__()

        #if mix_option:
        self.bert = BertModel4Mix.from_pretrained('bert')
        #else:
        #    self.bert = BertModel.from_pretrained('bert-base-uncased')

        #self.linear = nn.Sequential(nn.Linear(768, 128),
        #                            nn.Tanh(),
        #                            nn.Linear(128, num_labels))

    def forward(self, x,attention_mask=None,token_type_ids=None, x2=None,attention_mask_1=None,token_type_ids_1=None, l=None, mix_layer=10):

        if x2 is not None:
            all_hidden, pooler = self.bert(x,attention_mask,token_type_ids, x2,attention_mask_1,token_type_ids_1, l, mix_layer)

            pooled_output = torch.mean(all_hidden, 1)

        else:
            all_hidden, pooler = self.bert(x,attention_mask=attention_mask, token_type_ids=token_type_ids)

            pooled_output = torch.mean(all_hidden, 1)

        #predict = self.linear(pooled_output)

        return pooled_output

