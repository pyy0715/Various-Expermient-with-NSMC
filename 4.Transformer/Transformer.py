# Transfomer모델은 model.py에 구현되있습니다
import os
import sys
import time
import math
import numpy as np
from Korpora import Korpora, NSMCKorpus
from tokenizers import SentencePieceBPETokenizer
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from model import Encoder

sys.path.append('.')
from callbacks import seed_everything
from train import Trainer
from tokenizer import train_tokenizer, load_tokenizer
from dataset import NSMC_Dataset, make_batch
from corpus import make_corpus


if __name__ == "__main__":
   # Setting
   SEED = 42
   EPOCHS = 30
   BATCH_SIZE = 64
   device = "cuda" if torch.cuda.is_available() else "cpu"
   seed_everything(SEED)
   

   # Model
   class Transformer_Classification(nn.Module):
      def __init__(self, n_vocab, d_model, n_heads, n_layers, dim_feedforward, n_class=2, dropout=0.5):
         super(Transformer_Classification, self).__init__()
         self.encoder = Encoder(n_vocab, n_layers, d_model, n_heads, dim_feedforward, dropout=dropout)
         self.gru = nn.GRU(input_size=d_model,
                           hidden_size=dim_feedforward,
                           num_layers=n_layers,
                           batch_first=True,
                           bidirectional=True)
         self.decoder = nn.Linear(dim_feedforward*2, n_class)
         self.dim_feedforward = dim_feedforward
         self.n_layers = n_layers

      def forward(self, src):
         enc_output, _ = self.encoder(src)
         h_0 = self._init_state(batch_size=enc_output.size(0))
         output, _ = self.gru(enc_output, h_0)
         h_t = output[:, -1, :]
         logit = self.decoder(h_t)
         return logit
      
      def _init_state(self, batch_size=1):
         weight = next(self.parameters()).data
         return weight.new(self.n_layers*2, batch_size, self.dim_feedforward).zero_()
      
      
   # Load Tokenizer & Make Dataset
   sentencepiece_tokenizer = load_tokenizer(mode='sentencepiece')
   trn_dataset = NSMC_Dataset(mode='train', tokenizer=sentencepiece_tokenizer)
   vid_dataset = NSMC_Dataset(mode='test', tokenizer=sentencepiece_tokenizer)

   trn_dataloader = torch.utils.data.DataLoader(trn_dataset,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                num_workers=4,
                                                pin_memory=True,
                                                collate_fn=make_batch)

   vid_dataloader = torch.utils.data.DataLoader(vid_dataset,
                                                batch_size=BATCH_SIZE,
                                                shuffle=False,
                                                num_workers=4,
                                                collate_fn=make_batch)

   # HyperParameter and Optimizer
   model = Transformer_Classification(n_vocab=sentencepiece_tokenizer.get_vocab_size(),
                                      d_model=64, 
                                      n_heads=4, 
                                      n_layers=2,
                                      dim_feedforward=256,
                                      n_class=2,
                                      dropout=0.2
                                      )
   optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

   # Train & Evaluate
   trainer = Trainer(model=model,
                     optimizer=optimizer,
                     device=device)

   for epoch in range(1, EPOCHS+1):
      trainer.train(trn_dataloader)
      val_loss, val_accuracy = trainer.evaluate(vid_dataloader)
      print("[Epoch: %d] Val Loss:%5.2f | Val Acc:%5.2f" %
            (epoch, val_loss, val_accuracy))
