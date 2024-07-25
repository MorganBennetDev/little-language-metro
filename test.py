from transformers import pipeline

import random

fill_mask = pipeline(
    'fill-mask',
    model = './models/reginald',
    tokenizer = './models/reginald-tokens'
)

def next_token(past, k = 20):
    tokens = fill_mask(past + '<mask>', top_k = k)

    total = 0.0

    for guess in tokens:
        total += guess['score']

    choice = random.uniform(0.0, total)

    for guess in tokens:
        choice -= guess['score']

        if choice <= 0:
            return guess
        
    return tokens[-1]


current = ''

while True:
    current = current + next_token(current)['token_str']
    print(current)
    input()