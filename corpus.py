import os
import numpy as np
from Korpora import Korpora, NSMCKorpus

def make_corpus():
    if not os.path.isdir('corpus'):
        os.makedirs('corpus')     
    corpus = Korpora.load('nsmc').train
    np.savetxt('./corpus/train.txt', corpus.get_all_texts(), delimiter=' ', fmt='%s')       
    print('The corpus file has been built.')

if __name__ == '__main__':
    make_corpus()
