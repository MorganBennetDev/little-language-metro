import torch
from torch.utils.data import Dataset

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from get_names import get_city_names

class CityDataset(Dataset):
    def __init__(self):
        tokenizer = ByteLevelBPETokenizer(
            './models/tokenton/vocab.json',
            './models/tokenton/merges.txt'
        )

        tokenizer._tokenizer.post_processor = BertProcessing(
            ('</s>', tokenizer.token_to_id('</s>')),
            ('<s>', tokenizer.token_to_id('<s>'))
        )

        tokenizer.enable_truncation(max_length = 512)

        _, names = get_city_names()

        self.examples = [ x.ids for x in tokenizer.encode_batch(names) ]

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return torch.tensor(self.examples[i])