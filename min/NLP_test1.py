### Natural Language Processing
## pip install transformers <- 필수
## pip install sentencepiece <- 필수
# Load model directly
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

'''
TensorFlow에서 YOLO와 함께 사용할 수 있는 Seq2Seq Transformer 모델은 다음과 같습니다.

* **YOLOSeq:** YOLOSeq는 YOLOv5와 Transformer를 결합한 모델입니다. YOLOSeq는 이미지에서 객체를 감지하고 각 객체의 설명을 생성하는 데 사용할 수 있습니다.
* **DETR:** DETR는 Transformer 기반 객체 감지 모델입니다. DETR은 이미지에서 객체를 감지하고 각 객체의 클래스 및 바운딩 박스를 예측하는 데 사용할 수 있습니다.
* **Sparse Attention Transformer:** Sparse Attention Transformer는 Transformer 모델의 확장 버전입니다. Sparse Attention Transformer는 이미지에서 객체를 감지하고 각 객체의 클래스 및 바운딩 박스를 예측하는 데 사용할 수 있습니다.

다음은 각 모델에 대한 간략한 설명입니다.

**YOLOSeq**

YOLOSeq는 YOLOv5와 Transformer를 결합한 모델입니다. YOLOSeq는 이미지에서 객체를 감지하고 각 객체의 설명을 생성하는 데 사용할 수 있습니다. YOLOSeq는 다음과 같이 작동합니다.

1. YOLOv5는 이미지에서 객체를 감지하고 각 객체의 바운딩 박스를 예측합니다.
2. Transformer는 각 객체의 바운딩 박스를 기반으로 객체의 설명을 생성합니다.

YOLOSeq는 이미지에서 객체를 감지하고 각 객체의 설명을 생성하는 데 효과적인 모델입니다.

**DETR**

DETR은 Transformer 기반 객체 감지 모델입니다. DETR은 이미지에서 객체를 감지하고 각 객체의 클래스 및 바운딩 박스를 예측하는 데 사용할 수 있습니다. DETR은 다음과 같이 작동합니다.

1. Transformer는 이미지를 인코딩합니다.
2. Transformer는 인코딩된 이미지를 기반으로 각 객체의 클래스 및 바운딩 박스를 예측합니다.

DETR은 이미지에서 객체를 감지하는 데 효과적인 모델입니다.

**Sparse Attention Transformer**

Sparse Attention Transformer는 Transformer 모델의 확장 버전입니다. Sparse Attention Transformer는 이미지에서 객체를 감지하고 각 객체의 클래스 및 바운딩 박스를 예측하는 데 사용할 수 있습니다. Sparse Attention Transformer는 다음과 같이 작동합니다.

1. Sparse Attention Transformer는 이미지를 인코딩합니다.
2. Sparse Attention Transformer는 인코딩된 이미지를 기반으로 각 객체의 클래스 및 바운딩 박스를 예측합니다.

Sparse Attention Transformer는 이미지에서 객체를 감지하는 데 효과적인 모델입니다.

**사용할 모델 선택**

사용할 모델은 특정 요구 사항에 따라 다릅니다. 다음은 모델을 선택할 때 고려해야 할 몇 가지 사항입니다.

* **정확도:** YOLOSeq는 이미지에서 객체를 감지하고 각 객체의 설명을 생성하는 데 가장 정확한 모델입니다.
* **속도:** DETR은 이미지에서 객체를 감지하는 데 가장 빠른 모델입니다.
* **메모리 사용량:** Sparse Attention Transformer는 이미지에서 객체를 감지하는 데 가장 적은 메모리를 사용하는 모델입니다.

다음은 각 모델의 장점과 단점입니다.

**YOLOSeq**

**장점:**

* 높은 정확도
* 다양한 객체를 감지할 수 있음

**단점:**

* 느린 속도
* 높은 메모리 사용량

**DETR**

**장점:**

* 빠른 속도
* 낮은 메모리 사용량

**단점:**

* 낮은 정확도
* 제한된 수의 객체만 감지할 수 있음

**Sparse Attention Transformer**

**장점:**

* 낮은 메모리 사용량

**단점:**

* 낮은 정확도
* 느린 속도

**결론**

TensorFlow에서 YOLO와 함께 사용할 수 있는 다양한 Seq2Seq Transformer 모델이 있습니다. 
사용할 모델은 특정 요구 사항에 따라 다릅니다.
'''