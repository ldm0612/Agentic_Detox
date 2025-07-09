from language_id import LanguageDetector
from translation import SwahiliTranslator
from sentiment import analyze_sentiment
from toxicity import analyze_toxicity
from detox import detoxify_text
from llm import get_llm
from config import MODEL_ID, DEVICE_MAP, MAX_NEW_TOKENS, 


def process_texts_rule_based(texts):
    """
    Rule-based workflow:
    - Detect language
    - Translate if not English
    - Sentiment analysis
    - Toxicity analysis
    - Detoxify if toxic
    """

    results = []
    
    llm = get_llm(MODEL_ID, device_map=DEVICE_MAP, max_new_tokens=MAX_NEW_TOKENS)
    language_detector = LanguageDetector()
    sw_translator = SwahiliTranslator()


    for idx, text in enumerate(texts):
        print(f"\nProcessing sentence {idx}: {text}")

        # Step 1: Language detection
        is_english = language_detector.detect_eng(text)
        translated_sentence = ""
        current_text = text

        if not is_english:
            print("Non-English text detected. Translating...")
            translated_sentence = sw_translator.translate(text)
            current_text = translated_sentence

        # Step 2: Sentiment analysis
        sentiment_result = analyze_sentiment(llm, current_text)
        sentiment_label = sentiment_result.get("sentiment_label", "unknown")
        sentiment_explanation = sentiment_result.get("explanation", "")

        # Step 3: Toxicity analysis
        toxicity_result = analyze_toxicity(llm, current_text)
        toxicity_label = toxicity_result.get("toxicity_label", "unknown")
        toxicity_explanation = toxicity_result.get("explanation", "")

        # Step 4: Detoxification if needed
        detoxed_sentence = ""
        if toxicity_label == "toxic":
            print("Toxicity detected. Detoxifying...")
            detox_result = detoxify_text(llm, current_text)
            detoxed_sentence = detox_result.get("detox_text", "")

        results.append({
            "id": idx,
            "original_sentence": text,
            "translated_sentence": translated_sentence,
            "sentiment_prediction": sentiment_label,
            "sentiment_prediction_explanation": sentiment_explanation,
            "toxicity_prediction": toxicity_label,
            "toxicity_prediction_explanation": toxicity_explanation,
            "detoxed_sentence": detoxed_sentence
        })

    return results
