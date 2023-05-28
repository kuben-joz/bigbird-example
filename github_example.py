import torch
from transformers import BigBirdForPreTraining, BigBirdConfig, BigBirdTokenizer
from transformers.data import DataCollatorForLanguageModeling
import time

batch_size = 4

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


pre_input = torch.load('example_batch_64.pt')
input = {}
for key, val in pre_input.items():
    input[key] = val[:batch_size].to(device)

print(input)

tokenizer = BigBirdTokenizer(vocab_file='tokenizer.model', unk_token='[UNK]', pad_token='[PAD]', \
            model_max_length=4096)


collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                           pad_to_multiple_of=4096,
                                           return_tensors='pt'
                                           )

bigbird_config = BigBirdConfig(pad_token_id=tokenizer.pad_token_id,
                               bos_token_id=tokenizer.bos_token_id,
                               eos_token_id=tokenizer.eos_token_id,
                               sep_token_id=tokenizer.sep_token_id,
                               vocab_size = 32000
                               )


model = BigBirdForPreTraining(bigbird_config)



model.to(device)

res = model(**input)
res.loss.backward()
print("wait started")
time.sleep(20)

