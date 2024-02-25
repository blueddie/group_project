## 나오긴 하는데...

# from transformers import BartTokenizer, TFBartModel, TFBartForConditionalGeneration

# tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
# model = TFBartModel.from_pretrained('facebook/bart-base')

# inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
# outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states)

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



# All PyTorch model weights were used when initializing TFBartModel.

# All the weights of TFBartModel were initialized from the PyTorch model.
# If your task is similar to the task the model of the checkpoint was trained on, 
# you can already use TFBartModel for predictions without further training.

# 2024-02-24 22:40:06.948939: I tensorflow/stream_executor/cuda/cuda_blas.cc:1786] 
# TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.


from transformers import BartTokenizer, TFBartForConditionalGeneration

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = TFBartForConditionalGeneration.from_pretrained('facebook/bart-base')

input_text = "This is a test sentence."
input_ids = tokenizer(input_text, return_tensors="tf").input_ids

output_text = "This is a response sentence."
output_ids = tokenizer(output_text, return_tensors="tf").input_ids

# 모델 예측
outputs = model(input_ids=input_ids, labels=output_ids)

# 예측 결과 출력
predicted_ids = outputs.logits.numpy()
predicted_sentence = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

print(predicted_sentence)

