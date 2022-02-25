from model import *
from dataloader import *
from util import *

class PretrainModelManager:
    
    def __init__(self, args, data):

        self.model = BertForModel( num_labels = data.num_labels)
        if args.freeze_bert_parameters:
            for name, param in self.model.bert.named_parameters():  
                param.requires_grad = False
                if "encoder.layer.11" in name or "pooler" in name:
                    param.requires_grad = True
                    
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id           
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        n_gpu = 1
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        self.num_train_optimization_steps = int(len(data.train_examples) / args.train_batch_size) * args.num_train_epochs
        
        self.optimizer = self.get_optimizer(args)
        
        self.best_eval_score = 0

    def eval(self, args, data):
        
        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.num_labels)).to(self.device)
        
        for batch in tqdm(data.eval_dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                _, logits = self.model(input_ids, segment_ids, input_mask, mode = 'eval')
                total_labels = torch.cat((total_labels,label_ids))
                total_logits = torch.cat((total_logits, logits[:,:-1]))
        
        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim = 1)
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)

        return acc


    def train(self, args, data):    

        wait = 0
        best_model = None
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(data.train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode = "train",args=args)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
            
            loss = tr_loss / nb_tr_steps
            print('pretrain_loss',loss)
            
            eval_score = self.eval(args, data)
            print('eval_score',eval_score)
            
            if eval_score > self.best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_eval_score = eval_score
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break
                
        self.model = best_model
        if args.save_model:
            self.save_model(args)

    def get_optimizer(self, args):

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                         lr = args.lr,
                         warmup = args.warmup_proportion,
                         t_total = self.num_train_optimization_steps)   
        return optimizer
    
    def save_model(self, args):

        if not os.path.exists(args.pretrain_dir):
            os.makedirs(args.pretrain_dir)
        self.save_model = self.model.module if hasattr(self.model, 'module') else self.model  

        model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model_config_file = os.path.join(args.pretrain_dir, CONFIG_NAME)
        torch.save(self.save_model.state_dict(), model_file)
        with open(model_config_file, "w") as f:
            f.write(self.save_model.config.to_json_string())

