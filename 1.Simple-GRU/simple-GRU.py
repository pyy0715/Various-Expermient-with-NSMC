from Korpora import Korpora, NSMCKorpus
from tokenizers import SentencePieceBPETokenizer
from tqdm.auto import tqdm, trange

import os
import sys 
import time
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

sys.path.append('.')
from corpus import make_corpus
from dataset import NSMC_Dataset, make_batch
from tokenizer import train_tokenizer, load_tokenizer
from train import Trainer
from callbacks import seed_everything

if __name__ == "__main__":
   # Setting
   SEED = 42
   EPOCHS = 10
   BATCH_SIZE = 64
   device = "cuda" if torch.cuda.is_available() else "cpu"
   seed_everything(SEED)

   # Model
   class SimpleGRU(nn.Module):
      def __init__(self, hidden_dim, n_layers, n_vocab, embed_dim, n_class=2, dropout_p=0):
         super(SimpleGRU, self).__init__()
         self.embed = nn.Embedding(n_vocab, embed_dim, padding_idx=0)
         self.gru = nn.GRU(input_size = embed_dim, 
                           hidden_size = hidden_dim, 
                           num_layers = n_layers, 
                           batch_first=True,
                           dropout=dropout_p)
         self.out = nn.Linear(hidden_dim, n_class)
         self.hidden_dim = hidden_dim
         self.n_layers = n_layers
         
      def forward(self, x):
         embedded = self.embed(x)
         h_0 = self._init_state(batch_size=x.size(0))
         output, _ = self.gru(embedded, h_0)
         h_t = output[:, -1, :]
         logit = self.out(h_t)  # [b, h] -> [b, c]
         return logit

      def _init_state(self, batch_size=1):
         weight = next(self.parameters()).data
         return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()

   # Load Tokenizer & Make Dataset
   sentencepiece_tokenizer = load_tokenizer(mode='sentencepiece')
   trn_dataset = NSMC_Dataset(mode='train', tokenizer=sentencepiece_tokenizer)
   vid_dataset = NSMC_Dataset(mode='test', tokenizer=sentencepiece_tokenizer)

   trn_dataloader = torch.utils.data.DataLoader(trn_dataset,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                num_workers=2,
                                                collate_fn=make_batch)

   vid_dataloader = torch.utils.data.DataLoader(vid_dataset,
                                                batch_size=BATCH_SIZE,
                                                shuffle=False,
                                                num_workers=0,
                                                collate_fn=make_batch)

   # HyperParameter and Optimizer
   model = SimpleGRU(hidden_dim=64,
                     n_layers=1,
                     n_vocab=sentencepiece_tokenizer.get_vocab_size(),
                     embed_dim=16, 
                     n_class=2, 
                     dropout_p=0.2)
   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

   # Train & Evaluate
   trainer = Trainer(model=model, 
                     optimizer=optimizer, 
                     device=device)
                     
   for epoch in range(1, EPOCHS+1):
      trainer.train(trn_dataloader)
      val_loss, val_accuracy = trainer.evaluate(vid_dataloader)
      print("[Epoch: %d] Val Loss:%5.2f | Val Acc:%5.2f" %(epoch, val_loss, val_accuracy))


