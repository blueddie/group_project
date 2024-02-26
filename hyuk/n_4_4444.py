import numpy as np
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf
import re


model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = TFGPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)

input_text = "'clock', 'car', 'stop sign', 'truck' "
input_ids = tokenizer.encode(input_text, return_tensors='tf')

max_length = 80

output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("생성 : ",  generated_text)