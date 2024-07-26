from transformers import pipeline

generator = pipeline(model = './models/reginald', tokenizer = './models/reginald-tokens', task = 'text-generation', do_sample = True, max_new_tokens = 10)

print('\n'.join([ '* ' + generator(letter)[0]['generated_text'] for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' ]))