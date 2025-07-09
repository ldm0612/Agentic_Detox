from transformers import pipeline
import torch

class SwahiliTranslator:
    def __init__(self):
        self.translator = pipeline(
            "translation",
            model="UBC-NLP/toucan-base",
            device_map="auto",
            torch_dtype=torch.float16
        )

    def translate(self, text, max_length=256):
        return self.translator(text, max_length=max_length)[0]["translation_text"]


