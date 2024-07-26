from get_names import get_city_names

_, names = get_city_names()

# Tokenize names
from tokenizers import ByteLevelBPETokenizer
from tokenizers import normalizers
from tokenizers.normalizers import NFKC, Lowercase, StripAccents

tokenizer = ByteLevelBPETokenizer()

tokenizer.normalizer = normalizers.Sequence([ NFKC(), Lowercase(), StripAccents() ])

from tokenizers.processors import TemplateProcessing

tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2),
    ],
)

tokenizer.train_from_iterator(
    names,
    vocab_size = 1_000,
    min_frequency = 5,
    special_tokens = [
        '<s>',
        '<pad>',
        '</s>',
        '<unk>',
        '<mask>'
    ]
)

tokenizer.save_model('./models/tokenton')
