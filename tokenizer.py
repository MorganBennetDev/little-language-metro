from get_names import get_city_names

_, names = get_city_names()

# Tokenize names
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()

tokenizer.train_from_iterator(
    names,
    vocab_size = 20_000,
    min_frequency = 2,
    special_tokens = [
        '<s>',
        '<pad>',
        '</s>',
        '<unk>',
        '<mask>'
    ]
)

tokenizer.save_model('./models/tokenton')