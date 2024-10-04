import torch
import copy
import torch.nn.functional as F
import transformers
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import (AutoConfig,
                          AutoModelForSequenceClassification,
                          AutoTokenizer,
                          EvalPrediction,
                          HfArgumentParser,
                          PretrainedConfig,
                          Trainer,
                          TrainingArguments,
                          default_data_collator,
                          set_seed,
                          RobertaForSequenceClassification,
                          T5ForConditionalGeneration,
                          T5Tokenizer
                          )

from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
import json
import ast

class WarmupQualityEstimatorModel(pl.LightningModule):
    def __init__(self, model, learning_rate, coef, batch_size, loss_func, lr_control):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.coef = coef
        self.loss_func = loss_func
        self.lr_control = lr_control
        self.weight = 0.125
        
        if isinstance(self.model, RobertaForSequenceClassification):
            classifier_layers = ['classifier.dense.weight', 'classifier.out_proj.weight', 'classifier.out_proj.bias', 'classifier.dense.bias']
            
            self.classifier_parameters = [param for name, param in model.named_parameters() if name in classifier_layers]
            self.base_parameters = [param for name, param in model.named_parameters() if name not in classifier_layers]
        
    def forward(self, input_ids, attention_mask, label=None, decoder_input_ids=None):
        self.model.requires_grad_(requires_grad=True)
        
        if isinstance(self.model, RobertaForSequenceClassification):            
            output = self.model(
                input_ids = input_ids, 
                attention_mask = attention_mask
                )        
            return output.logits
        else:
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=label, decoder_input_ids = decoder_input_ids, return_dict=True)
            
            return output.loss, output.logits
    
    def training_step(self, batch, batch_idx, optimizer_idx = None):
        
        context_input_ids = batch['context_input_ids'] 
        context_attention_mask = batch['context_attention_mask'] 
        label = batch['label'] 
        decoder_input_ids = batch['decoder_input_ids']
        batch_size = context_input_ids.shape[0]
        
        if self.lr_control == "sep":
            if optimizer_idx == 0:
                logits = self(
                    input_ids = context_input_ids, 
                    attention_mask = context_attention_mask
                    )   
                
                loss = F.cross_entropy(logits, label)
               
                
            elif optimizer_idx == 1:
                logits = self(
                    input_ids = context_input_ids, 
                    attention_mask = context_attention_mask
                    )  
                
                loss = F.cross_entropy(logits, label)
        
        else:
            if isinstance(self.model, RobertaForSequenceClassification):
                logits = self(
                    input_ids = context_input_ids, 
                    attention_mask = context_attention_mask
                    )   
                
                loss = F.cross_entropy(logits, label)       
            else:
                loss_weight = copy.deepcopy(label[:,0])
                loss_weight = loss_weight.type(torch.float)
                loss_weight[loss_weight==6136] = self.weight 
                loss_weight[loss_weight==1176] = 1
                
                loss, logit = self(context_input_ids, context_attention_mask, label) # decoder_input_ids 이거 제외했는데 나중에 확인
                
                criterion = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                logits_reshaped = logit.view(-1, 32128)
                labels_reshaped = label.view(-1)
                loss = criterion(logits_reshaped, labels_reshaped)
                loss = loss.view(batch_size, 15) # have to sync target_max_length
                batch_losses = loss.mean(dim=1)
                
                loss = loss_weight * batch_losses
                loss = loss.sum()
                    
                
        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=self.batch_size)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        context_input_ids = batch['context_input_ids'] # (batch, seq_len)
        context_attention_mask = batch['context_attention_mask'] # (batch, seq_len)
        label = batch['label'] # (batch, 1)
        #label = label.squeeze(1)
        decoder_input_ids = batch['decoder_input_ids'] 
        batch_size = context_input_ids.shape[0]
        
        if isinstance(self.model, RobertaForSequenceClassification):
            logits = self(
                input_ids = context_input_ids, 
                attention_mask = context_attention_mask
                ) # (batch, 2)    
            
            loss = F.cross_entropy(logits, label)       
        else:
            loss, logit = self(context_input_ids, context_attention_mask, label)

        self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=self.batch_size)
        
        return loss
    
    def test_step(self, batch, batch_idx):        
        
        context_input_ids = batch['context_input_ids'] 
        context_attention_mask = batch['context_attention_mask']
        label = batch['label'] 
        decoder_input_ids = batch['decoder_input_ids'] 
        
        if isinstance(self.model, RobertaForSequenceClassification):
            logits = self(
                input_ids = context_input_ids, 
                attention_mask = context_attention_mask
                ) # (batch, 2)    
            
            loss = F.cross_entropy(logits, label)       
        else:
            loss, logit = self(context_input_ids, context_attention_mask, label, decoder_input_ids)            
        
        self.log("test_loss", loss, prog_bar=True, logger=True, batch_size=self.batch_size)
        
        return loss
    
    def cal_acc(self, label, probs):
        
        above_threshold_indices = torch.where(probs >= 0.5)[0]
        target_index = torch.where(label == 1)[0]
        
        if len(above_threshold_indices) == len(target_index) and torch.all(above_threshold_indices == target_index):
            return torch.tensor([1.0], requires_grad=True).to(probs.device)
        else:
            return torch.tensor([0.0], requires_grad=True).to(probs.device)
    
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-3)


class WarmupQuestionContextDataset(Dataset):
    def __init__(self, data, tokenizer, model, source_max_token_len=256, target_max_token_len = 5):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data
        self.model = model
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        
        question = data_row.question
        ctxs = data_row.ctxs    
        
        if isinstance(self.tokenizer, T5TokenizerFast):
            doc = "question: " + question + "context: " + ctxs['title'] + ". " + ctxs['text']
            decoder_input_ids = self.tokenizer("<pad>", return_tensors="pt").input_ids[:,0].flatten()
        else:
            doc = "<s> " + question + "</s> <s>" + ctxs['title'] + "." + ctxs['text'] + " </s>"
            decoder_input_ids = self.tokenizer("blank", return_tensors="pt").input_ids[:,0].unsqueeze(0)
            
        if isinstance(self.tokenizer, T5TokenizerFast):            
            if ctxs['has_answer'] == True:
                _label = "true"
            else:
                _label = "false"   
            _label = self.tokenizer(
                _label,
                return_tensors="pt",
                padding="max_length",  
                max_length=self.target_max_token_len,  
                add_special_tokens=True,
                truncation=True,
                )
            label = _label['input_ids'].flatten()
            label[label == 0] = -100
        else:
            if ctxs['has_answer'] == True:
                label = torch.tensor([1])
            else:
                label = torch.tensor([0])        
        

        inputs = self.tokenizer(
            doc,
            return_tensors="pt",
            padding="max_length",  
            max_length=self.source_max_token_len,  
            add_special_tokens=True,
            truncation=True,
        )
        context_input_ids = inputs['input_ids'].flatten()
        context_attention_mask = inputs['attention_mask'].flatten()
        
        
        
        return {"context_input_ids" : context_input_ids,
                "context_attention_mask" : context_attention_mask,
                "label" : label} 
    


class WarmupQuestionContextModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, model, tokenizer, batch_size=8, source_max_token_len=256, target_max_token_len = 5):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.model = model

    def setup(self):
        self.train_dataset = WarmupQuestionContextDataset(self.train_df, self.tokenizer, self.model,
                                             self.source_max_token_len, self.target_max_token_len)
        self.test_dataset = WarmupQuestionContextDataset(self.test_df, self.tokenizer, self.model,
                                            self.source_max_token_len, self.target_max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=12)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=12)