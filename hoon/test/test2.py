import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# MS-COCO 데이터셋 클래스 정의
class CocoDataset:
    def __init__(self, json_file, image_folder):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.image_folder = image_folder

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        img_name = self.data['images'][idx]['file_name']
        img_id = self.data['images'][idx]['id']
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert('RGB')

        captions = []
        for ann in self.data['annotations']:
            if ann['image_id'] == img_id:
                captions.append(ann['caption'])

        return image, captions

def preprocess_caption(caption):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(caption.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# 하이퍼파라미터 설정
vocab_size = 10000  # 어휘 사전의 크기
embedding_dim = 256  # 임베딩 차원
lstm_units = 512  # LSTM의 은닉 상태 크기
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# 데이터셋과 데이터 제너레이터 준비
dataset = CocoDataset(json_file='C:\\group_project_data\\coco2017\\annotations\\captions_train2017.json', image_folder='C:\\group_project_data\\coco2017\\train2017\\train')

# 텍스트 토큰화를 위한 Tokenizer 객체 생성
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<unk>")
# 캡션 텍스트를 토큰화하여 토큰 사전 생성
all_captions = []
for _, captions in dataset:
    all_captions.extend(captions)
tokenizer.fit_on_texts(all_captions)

# 모델 생성
class ImageCaptioningModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, lstm_units):
        super(ImageCaptioningModel, self).__init__()
        self.image_feature_extractor = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
        self.image_feature_extractor.trainable = False
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, img_inputs, caption_inputs):
        features = self.image_feature_extractor(img_inputs)
        features = tf.keras.layers.GlobalAveragePooling2D()(features)
        caption_embeddings = self.embedding(caption_inputs)
        inputs = tf.concat([tf.expand_dims(features, 1), caption_embeddings], axis=1)
        lstm_outputs = self.lstm(inputs)
        outputs = self.dense(lstm_outputs)
        return outputs

# 손실 함수 및 최적화 알고리즘 설정
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
optimizer = tf.keras.optimizers.Adam(learning_rate)

# 모델 훈련
@tf.function
def train_step(img_inputs, caption_inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(img_inputs, caption_inputs)
        loss = loss_object(target, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

model = ImageCaptioningModel(vocab_size, embedding_dim, lstm_units)

for epoch in range(num_epochs):
    total_loss = 0
    for batch, (img_inputs, captions) in enumerate(dataset):
        caption_inputs = [preprocess_caption(caption) for caption in captions]
        caption_inputs = tokenizer.texts_to_sequences(caption_inputs)
        caption_inputs = pad_sequences(caption_inputs, padding='post')
        target = caption_inputs[:, 1:]
        caption_inputs = caption_inputs[:, :-1]
        loss = train_step(img_inputs, caption_inputs, target)
        total_loss += tf.reduce_sum(loss)
        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, loss.numpy().mean()))
    print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / len(dataset)))
