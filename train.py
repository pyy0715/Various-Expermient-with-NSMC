import torch 
import torch.nn.functional as F
from tqdm import tqdm

class Trainer(object):
   def __init__(self, model, optimizer, device):
      self.model = model.to(device)
      self.optimizer = optimizer
      self.device = device

   def train(self, trn_iter):
      """train model"""
      self.model.train()  # train mode
      for idx, batch in enumerate(tqdm(trn_iter)):
         x, y = batch['input'].to(self.device), batch['label'].to(self.device)
         self.optimizer.zero_grad()
         logit = self.model(x)
         loss = F.cross_entropy(logit, y)
         loss.backward()
         self.optimizer.step()

   def evaluate(self, val_iter):
      """evaluate model"""
      self.model.eval()  # train mode
      corrects, total_loss = 0, 0
      for idx, batch in enumerate(tqdm(val_iter)):
         x, y = batch['input'].to(self.device), batch['label'].to(self.device)
         logit = self.model(x)
         loss = F.cross_entropy(logit, y, reduction='sum')
         total_loss += loss.item()
         pred = logit.max(1, keepdim=True)[1]
         corrects += pred.eq(y.view_as(pred)).sum().item()

      size = len(val_iter.dataset)
      avg_loss = total_loss / size
      avg_accuracy = 100.0 * corrects / size
      return avg_loss, avg_accuracy
