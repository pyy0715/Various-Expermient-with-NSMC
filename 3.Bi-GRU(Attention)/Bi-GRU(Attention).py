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
   EPOCHS = 30
   BATCH_SIZE = 64
   device = "cuda" if torch.cuda.is_available() else "cpu"
   seed_everything(SEED)

   # Model
   class BiGRU_Attention(nn.Module):
      def __init__(self, hidden_dim, n_layers, n_vocab, embed_dim, n_classes=2, dropout_p=0):
         super(BiGRU_Attention, self).__init__()
         self.embed = nn.Embedding(n_vocab, embed_dim, padding_idx=0)
         self.gru = nn.GRU(input_size=embed_dim,
                           hidden_size=hidden_dim,
                           num_layers=n_layers,
                           batch_first=True,
                           dropout=dropout_p,
                           bidirectional=True)
         self.out = nn.Linear(hidden_dim*2, n_classes)
         self.hidden_dim = hidden_dim
         self.n_layers = n_layers

      def attention_net(self, gru_output, final_state):
         # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
         hidden = final_state.view(-1, self.hidden_dim * 2, self.n_layers)
         
         if self.n_layers > 1:
            hidden = torch.mean(hidden, axis=-1).unsqueeze(2)
            attn_weights = torch.bmm(gru_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
         else:
            attn_weights = torch.bmm(gru_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
            
         soft_attn_weights = torch.nn.functional.softmax(attn_weights, 1)
         # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
         context = torch.bmm(gru_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        # context : [batch_size, n_hidden * num_directions(=2)]
         return context, soft_attn_weights.data.cpu().numpy()

      def forward(self, x):
         embedded = self.embed(x)
         h_0 = self._init_state(batch_size=x.size(0))
         output, hidden_state = self.gru(embedded, h_0)  # [i, b, h]
         attn_output, attention = self.attention_net(output, hidden_state)
         logit = self.out(attn_output)  # [b, h] -> [b, c]
         return logit

      def _init_state(self, batch_size=1):
         weight = next(self.parameters()).data
         return weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_()

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
   model = BiGRU_Attention(hidden_dim=64,
                           n_layers=2,
                           n_vocab=sentencepiece_tokenizer.get_vocab_size(),
                           embed_dim=16,
                           n_classes=2,
                           dropout_p=0.2)
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

   # Train & Evaluate
   trainer = Trainer(model=model,
                     optimizer=optimizer,
                     device=device)

   for epoch in range(1, EPOCHS+1):
      trainer.train(trn_dataloader)
      val_loss, val_accuracy = trainer.evaluate(vid_dataloader)
      print("[Epoch: %d] Val Loss:%5.2f | Val Acc:%5.2f" %
            (epoch, val_loss, val_accuracy))
