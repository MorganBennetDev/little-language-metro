from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size = 20_000,
    max_position_embeddings = 514,
    num_attention_heads = 12,
    num_hidden_layers = 6,
    type_vocab_size = 1
)

from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained('./models/tokenton', max_len = 512)

tokenizer.save_pretrained('./models/reginald-tokens')

from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config = config)

from dataset import CityDataset
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer, mlm = True, mlm_probability = 0.15
)

from transformers import Trainer, TrainingArguments
import os

os.environ['WANDB_PROJECT'] = 'little-language-metro'
os.environ['WANDB_LOG_MODEL'] = 'checkpoint'

training_args = TrainingArguments(
    output_dir = './models/reginald',
    overwrite_output_dir = True,
    num_train_epochs = 1,
    per_device_train_batch_size = 64,
    save_steps = 10_000,
    save_total_limit = 2,
    prediction_loss_only = True,
    report_to = 'wandb',
    logging_steps = 5
)

trainer = Trainer(
    model = model,
    args = training_args,
    data_collator = data_collator,
    train_dataset = CityDataset()
)

trainer.train()

trainer.save_model('./models/reginald')