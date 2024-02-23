### Natural Language Processing
## pip install sentencepiece <- 필수
# Load model directly
import torch
from transformers import AutoTokenizer, TFAutoModel,AutoModelForSeq2SeqLM
# Use a pipeline as a high-level helper
from transformers import pipeline

model = TFAutoModel.from_pretrained("google-bert/bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

# pipe = pipeline("text2text-generation", model="google/flan-t5-base")
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state

# print(last_hidden_states)
# 출력을 토큰 ID에서 텍스트로 변환
decoded_output = tokenizer.decode(outputs.logits[0], skip_special_tokens=True)

# 출력을 텍스트로 출력
print(decoded_output)


'''
# ============================================================
inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
outputs = model(inputs)

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states)


# ============================================================
'''


# # 입력 문장 토큰화
# input_ids = tokenizer.encode("This is a test sentence.", return_tensors="tf")

# # 출력 문장 토큰화
# output_ids = tokenizer.encode("This is a response sentence.", return_tensors="tf")

# # 모델 예측
# outputs = model(input_ids=input_ids, labels=output_ids)

# # 예측 결과 출력
# predicted_tokens = outputs.logits.argmax(dim=-1)
# predicted_sentence = tokenizer.decode(predicted_tokens)

# print(predicted_sentence)
