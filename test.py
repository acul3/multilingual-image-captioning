from models import FlaxCLIPVisionMarianForConditionalGeneration
from models import FlaxCLIPVisionMBartForConditionalGeneration
flax_clip_vision_marian_cg = FlaxCLIPVisionMarianForConditionalGeneration.from_clip_vision_marian_pretrained('openai/clip-vit-base-patch32','Helsinki-NLP/opus-mt-en-id')
#flax_clip_vision_mbart_cg = FlaxCLIPVisionMBartForConditionalGeneration.from_clip_vision_mbart_pretrained('openai/clip-vit-base-patch32', 'facebook/mbart-large-50')