from transformers import pipeline

fill_mask = pipeline(
    'fill-mask',
    model = './models/reginald',
    tokenizer = './models/reginald-tokens'
)

print(fill_mask(input() + "<mask>"))