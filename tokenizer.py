# Extract names
import pandas as pd

cities_df = pd.read_csv('./data/cities.csv', sep = ';', encoding = 'utf8')

datasets = { }
big_data = [ ]

print('Extracting city names')

for _, row in cities_df.iterrows():
    country = row['Country name EN']
    city = row['Name']

    if not isinstance(city, str) or not isinstance(country, str):
        continue

    big_data.append(city)

    if country in datasets:
        datasets[country].append(city)
    else:
        datasets[country] = [ city ]

# Tokenize names
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()

tokenizer.train_from_iterator(
    big_data,
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

tokenizer.save_model('./tokenton', 'tokenton')