import requests
from PIL import Image
from transformers import BlipProcessor, TFAutoModelForConditionalGeneration
import tensorflow as tf

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = TFAutoModelForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
# img_url = 'c:/_data/image/horse_human/horses/horse03-7.png'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# conditional image captioning
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="tf")

out = model.generate(inputs, max_length=50)  # max_length 값은 출력되는 문장의 최대 길이를 설정합니다.
decoded_output = processor.decode(out[0], skip_special_tokens=True)
print(decoded_output)
# >>> a photography of a woman and her dog

# unconditional image captioning
inputs = processor(raw_image, return_tensors="tf")

out = model.generate(inputs, max_length=50)  # max_length 값은 출력되는 문장의 최대 길이를 설정합니다.
decoded_output = processor.decode(out[0], skip_special_tokens=True)
print(decoded_output)
# >>> a woman sitting on the beach with her dog
