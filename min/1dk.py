import requests
from PIL import Image
from transformers import BlipProcessor, TFBlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = TFBlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 두 줄 세트
# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# 두 줄 세트
img_path = 'C:/_data/image/horse_human/horses/horse03-7.png'
raw_image = Image.open(img_path).convert('RGB')

# conditional image captioning
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="tf")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
# >>> a photography of a woman and her dog

# unconditional image captioning
inputs = processor(raw_image, return_tensors="tf")

out = model.generate(**inputs)
print("="*50)
print(processor.decode(out[0], skip_special_tokens=True))
# >>> a woman sitting on the beach with her dog