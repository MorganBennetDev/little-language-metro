from transformers import AutoTokenizer

prompt = input('Start of name: ')

tokenizer = AutoTokenizer.from_pretrained('./models/reginald-tokens')
inputs = tokenizer(prompt, return_tensors = 'pt').input_ids

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('./models/reginald')
outputs = model.generate(inputs, max_new_tokens = 10, do_sample = True, top_k = 50, top_p = 0.5, num_beams = 3, eos_token_id = model.config.eos_token_id)

print(tokenizer.batch_decode(outputs, skip_special_tokens = True))
