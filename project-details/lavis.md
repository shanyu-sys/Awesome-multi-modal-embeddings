
### LAVIS - A Library for Language-Vision Intelligence

#### Code

* [Github](https://github.com/salesforce/LAVIS/tree/main)
* Stars: **6k**
* License: BSD-3-Clause license

#### paper

A library with multiple papers and models including

* **InstructBLIP**: [preprint](https://arxiv.org/abs/2305.06500)
* **BLIP-2**: [ICML 2023](https://arxiv.org/abs/2301.12597)
* **Img2LLM-VQA**: [CVPR 2023](https://arxiv.org/pdf/2212.10846.pdf)

#### Installation and get embeddings

##### Installation: 

```
pip install salesforce-lavis
```

##### get embeddings: 

[demo](https://github.com/salesforce/LAVIS/blob/3446bac20c5646d35ae383ebe6d13cec4f8b00cb/examples/blip2_feature_extraction.ipynb)

```python
import torch
from PIL import Image

from lavis.models import load_model_and_preprocess

def get_image_text_embeddings(image, caption):
  model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
	image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
	text_input = txt_processors["eval"](caption)
	sample = {"image": image, "text_input": [text_input]}
	features_image = model.extract_features(sample, mode="image")
	features_text = model.extract_features(sample, mode="text")
	print(features_image.image_embeds.shape)
	# torch.Size([1, 32, 768])
	print(features_text.text_embeds.shape)
	# torch.Size([1, 12, 768])
  return featuers_image, features_text

raw_image = Image.open("../docs/_static/merlion.png").convert("RGB")
caption = "a large fountain spewing water into the air"
```