import os 
from tokenizers import SentencePieceBPETokenizer, BertWordPieceTokenizer
from tokenizers import normalizers
from transformers import BertTokenizer

def train_tokenizer(corpus, vocab_size, frequency, mode):
   if not os.path.isdir('tokenizer'):
       os.makedirs('tokenizer')
       
   # Sentencepiece
   if mode=='sentencepiece':
      tokenizer = SentencePieceBPETokenizer(add_prefix_space=True)
      tokenizer.normalize = normalizers.NFKC()
      tokenizer.train(
         files = corpus,
         vocab_size = vocab_size,
         min_frequency = frequency,
         special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
         show_progress=True,
      )
      tokenizer.save_model('./tokenizer', 'nsmc_sentencepiece')
      
   # Bert WordPiece
   elif mode=='bert':
      tokenizer = BertWordPieceTokenizer(lowercase=True)
      tokenizer.normalize = normalizers.NFKC()
      tokenizer.train(
         files = corpus,
         vocab_size = vocab_size,
         min_frequency = frequency,
         special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
         show_progress=True,
         )
      tokenizer.save_model('./tokenizer', 'nsmc_bert')
   print('Tokenizer training is completed and saved.\n')

def load_tokenizer(mode):
   if mode=='sentencepiece':
      tokenizer = SentencePieceBPETokenizer(
         vocab='./tokenizer/nsmc_sentencepiece-vocab.json',
         merges='./tokenizer/nsmc_sentencepiece-merges.txt',
      )
   elif mode=='bert':
      tokenizer = BertTokenizer(
         vocab_file='./tokenizer/nsmc_bert-vocab.txt'
      )
   return tokenizer


if __name__ == '__main__':
   corpus = './corpus/train.txt'
   vocab_size = 32000
   frequency = 5
   
   train_tokenizer(corpus, vocab_size, frequency, 'sentencepiece')
   train_tokenizer(corpus, vocab_size, frequency, 'bert')
