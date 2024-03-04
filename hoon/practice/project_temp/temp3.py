import numpy as np

# 예시 데이터셋 생성
X = np.array([[1, 2, 3],  # 첫 번째 샘플
              [4, 5, 6],  # 두 번째 샘플
              [7, 8, 9]]) # 세 번째 샘플

# 배치 정규화(Batch Normalization)
mean = np.mean(X, axis=0)  # 각 특성별 평균 계산
std = np.std(X, axis=0)    # 각 특성별 표준편차 계산
X_batch_normalized = (X - mean) / std  # 배치 정규화
print(X_batch_normalized)

# 레이어 정규화(Layer Normalization)
mean = np.mean(X, axis=1, keepdims=True)  # 각 샘플별 평균 계산
std = np.std(X, axis=1, keepdims=True)    # 각 샘플별 표준편차 계산
X_layer_normalized = (X - mean) / std  # 레이어 정규화
print(X_layer_normalized)