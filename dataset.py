from Korpora import Korpora, NSMCKorpus
import torch
from torch.utils.data import Dataset

class NSMC_Dataset(Dataset):
    def __init__(self, mode, tokenizer):
        self.mode = mode
        self.tokenizer = tokenizer
        self.nsmc = Korpora.load('nsmc').train if self.mode == 'train' else Korpora.load('nsmc').test

    def __len__(self):
        return len(self.nsmc)

    def __getitem__(self, index):
        text = self.nsmc[index].text
        label = self.nsmc[index].label
        encode = self.tokenizer.encode(text).ids
        return ({
            'input': torch.tensor(encode, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        })

# Padding and Sorting
def make_batch(samples):
    inputs = [sample['input'] for sample in samples]
    labels = [sample['label'] for sample in samples]
    lengths = torch.Tensor([len(sample['input']) for sample in samples])
    input_lengths, sorted_idx = lengths.sort(0, descending=True)

    padded_inputs = torch.nn.utils.rnn.pad_sequence(
        inputs, batch_first=True, padding_value=0)

    return {'input': padded_inputs.contiguous()[sorted_idx],
            'label': torch.stack(labels).contiguous()[sorted_idx],
            }


