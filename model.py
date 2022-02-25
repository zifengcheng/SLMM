from util import *
from mixtext import *

class BertForModel(nn.Module):
    def __init__(self,num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.bert = MixText()
        self.dense = nn.Linear(768, 768)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768,num_labels)
        self.classifier_2 = nn.Linear(768,3)


    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None,input_ids_1 = None, token_type_ids_1 = None, attention_mask_1=None , labels_1 = None,mask = None,l = None,args=None):
        encoded_layer_12 = self.bert(input_ids,  attention_mask,token_type_ids)

        pooled_output = self.dense(encoded_layer_12)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits_none = self.classifier_2(pooled_output).max(1).values.unsqueeze(1)
        prediction = torch.cat((logits,logits_none),1)
        #print('max is ',logits.max(1).values,logits_none.squeeze(1),logits.max(1).values>logits_none.squeeze(1))
              
        if input_ids_1 != None:
            encoded_layer_12 = self.bert(input_ids,  attention_mask,token_type_ids,input_ids_1, token_type_ids_1, attention_mask_1,l)
            pooled_output = self.dense(encoded_layer_12)
            pooled_output = self.activation(pooled_output)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            logits_none = self.classifier_2(pooled_output).max(1).values.unsqueeze(1)
            prediction1 = torch.cat((logits,logits_none),1)
        
        if feature_ext:
            return pooled_output
        else:
            if mode == 'train':
                loss_1 = nn.CrossEntropyLoss()(logits,labels)
                labels_ = torch.zeros(prediction.shape).cuda()
                for i,j in enumerate(labels):
                    labels_[i,j] = 1 - args.beta
                labels_[:,-1] = args.beta
                #print(prediction.shape,labels_.shape)
                #loss = nn.KLDivLoss(reduction='batchmean')(prediction.softmax(dim=-1).log(),labels_.softmax(dim=-1))
                loss = nn.KLDivLoss(reduction='batchmean')(prediction.softmax(dim=-1).log(),labels_)
                if input_ids_1 == None:
                    return loss_1                
                loss_2 = nn.CrossEntropyLoss()(prediction1[mask],labels_1[mask])

                return  args.gamma * loss  + (1 - args.gamma) * loss_2
            else:
                return pooled_output, prediction