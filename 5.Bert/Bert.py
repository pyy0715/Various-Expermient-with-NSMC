import os
import sys
import time
import math
import numpy as np
from tqdm import tqdm

from Korpora import Korpora, NSMCKorpus

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import BertTokenizer, BertConfig, BertModel, AdamW, get_linear_schedule_with_warmup

sys.path.append('.')
from tokenizer import train_tokenizer, load_tokenizer
from callbacks import seed_everything


class Review_Dataset(Dataset):
   def __init__(self, mode, tokenizer):
      self.mode = mode
      self.tokenizer = tokenizer
      self.nsmc = Korpora.load('nsmc').train if self.mode == 'train' else Korpora.load('nsmc').test
      
   def __len__(self):
      return len(self.nsmc)

   def __getitem__(self, index):
      text = self.nsmc[index].text
      label = self.nsmc[index].label
      encoding = self.tokenizer.encode_plus(
         text,
         add_special_tokens=True,
         truncation=True,
         max_length=158, # NSMC Korpus's Max Length 
         return_token_type_ids=False,
         padding='max_length',
         return_attention_mask=True,
         return_tensors='pt',
         )
      return {
         'input_ids': encoding['input_ids'].flatten(),
         'attention_mask': encoding['attention_mask'].flatten(),
         'targets': torch.tensor(label, dtype=torch.long)
      }
      

class Sentimental_Classifer(nn.Module):
   def __init__(self, n_class, config):
      super(Sentimental_Classifer, self).__init__()
      self.bert = BertModel(config)
      self.drop = nn.Dropout(p=0.3)
      self.out = nn.Linear(self.bert.config.hidden_size, n_class)

   def forward(self, input_ids, attention_mask):
      _, pooled_output = self.bert(
         input_ids=input_ids,
         attention_mask=attention_mask,
         return_dict=False
      )
      output = self.drop(pooled_output)
      return self.out(output)
   

class Trainer(object):
   def __init__(self, model, optimizer, loss_fn, scheduler, device):
      self.model = model
      self.optimizer = optimizer
      self.loss_fn = loss_fn
      self.scheduler = scheduler
      self.device = device

   def train(self, trn_iter):
      """train model"""
      corrects, total_loss = 0, 0
      self.model.train()  # train mode
      for idx, batch in enumerate(tqdm(trn_iter)):
         input_ids = batch['input_ids'].to(device)
         attn_mask = batch["attention_mask"].to(device)
         targets = batch["targets"].to(device)

         self.optimizer.zero_grad()

         logit = self.model(input_ids, attn_mask)
         pred = logit.max(1, keepdim=True)[1]

         loss = self.loss_fn(logit, targets)
         total_loss += loss.item()

         loss.backward()
         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
         
         self.optimizer.step()
         self.scheduler.step()

         corrects += pred.eq(targets.view_as(pred)).sum().item()

      size = len(trn_iter.dataset)
      avg_loss = total_loss / size
      avg_accuracy = 100.0 * corrects / size
      return avg_loss, avg_accuracy

   def evaluate(self, val_iter):
      """evaluate model"""
      corrects, total_loss = 0, 0
      self.model.eval()  # eval mode
      for idx, batch in enumerate(tqdm(val_iter)):
         input_ids = batch['input_ids'].to(device)
         attn_mask = batch["attention_mask"].to(device)
         targets = batch["targets"].to(device)

         self.optimizer.zero_grad()

         logit = self.model(input_ids, attn_mask)
         pred = logit.max(1, keepdim=True)[1]

         loss = self.loss_fn(logit, targets)
         total_loss += loss.item()

         corrects += pred.eq(targets.view_as(pred)).sum().item()

      size = len(val_iter.dataset)
      avg_loss = total_loss / size
      avg_accuracy = 100.0 * corrects / size
      return avg_loss, avg_accuracy



if __name__ == "__main__":
   # Setting
   SEED = 42
   EPOCHS = 5
   BATCH_SIZE = 64
   device = "cuda" if torch.cuda.is_available() else "cpu"
   seed_everything(SEED)
   
   bert_tokenizer = load_tokenizer(mode='bert')

   trn_dataset = Review_Dataset(mode='train', tokenizer=bert_tokenizer)
   vid_dataset = Review_Dataset(mode='test', tokenizer=bert_tokenizer)
   
   trn_dataloader = torch.utils.data.DataLoader(trn_dataset,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                num_workers=4)

   vid_dataloader = torch.utils.data.DataLoader(vid_dataset,
                                                batch_size=BATCH_SIZE,
                                                shuffle=False,
                                                num_workers=4)
   
   configuration = BertConfig(vocab_size=32000, num_hidden_layers=4, num_attention_heads=4)
   model = Sentimental_Classifer(n_class=2, config=configuration).to(device)
   
   optimizer = AdamW(model.parameters(), lr=3e-5)
   
   total_steps = len(trn_dataloader) * EPOCHS
   scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=0,
      num_training_steps=total_steps
   )

   loss_fn = nn.CrossEntropyLoss()

   Trainer = Trainer(model, optimizer, loss_fn, scheduler, device)
   
   for epoch in range(EPOCHS):
      trn_loss, trn_acc = Trainer.train(trn_dataloader)
      vid_loss, vid_acc = Trainer.evaluate(vid_dataloader)

      print(f'Epoch {epoch + 1}/{EPOCHS}')
      print('-' * 10)
      print(f'Train: \t loss {trn_loss:.6}, \t accuracy {trn_acc:.4}')
      print(f'Val: \t loss {vid_loss:.6}, \t accuracy {vid_acc:.4}')
