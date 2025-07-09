from transformers import pipeline

class LanguageDetector:
    def __init__(self):
        self.detector = pipeline(
            "text-classification",
            model="papluca/xlm-roberta-base-language-detection",
            device="cuda"
        )

    def detect_eng(self, text):
        label = self.detector(text)[0]["label"]
        return label == "en"

