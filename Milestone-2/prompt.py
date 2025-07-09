SENTIMENT_PROMPT = """
Analyze the sentiment of the text and output ONLY a flat JSON object with exactly these two keys:

- "sentiment_label": one of ["positive", "negative", "mixed", "neutral"]
- "explanation": a short explanation for the label

Do NOT include any extra fields, nested objects, or additional text. The response must be a valid JSON object.

TEXT: {text}

{format_instructions}
"""

TOXICITY_PROMPT = """
Analyze the toxicity of the text and return ONLY a flat JSON object with exactly these two keys:

- "toxicity_label": one of ["toxic", "non-toxic"]
- "explanation": a short explanation for the chosen label

Do NOT include any extra fields, nested objects, or additional text. Return valid JSON only.

TEXT: {text}

{format_instructions}
"""

DETOX_PROMPT = """
You will be given a toxic sentence. Rewrite the text to remove all offensive, harmful, or disrespectful language while preserving the original intent as much as possible. Output ONLY a flat JSON object with exactly these two keys:

- "detox_text": the rewritten, detoxified version of the original text (non-toxic and respectful).
- "explanation": a short explanation for how you modified the text.

Do NOT include any extra fields, nested objects, or additional text. The response must be a valid JSON object.

TEXT: {text}

{format_instructions}

"""

AGENT_PROMPT = """
You are a text-processing agent with access to the following tools:

- LanguageDetector: Detects whether a text is in English. Returns True if English, False otherwise.
- SwahiliTranslator: Translates Swahili text to English.
- SentimentAnalyzer: Analyzes sentiment of English text. Returns JSON with 'sentiment_label' and 'explanation'.
- ToxicityAnalyzer: Analyzes toxicity of English text. Returns JSON with 'toxicity_label' and 'explanation'.
- Detoxifier: Rewrites toxic text to remove offensive language while preserving meaning. Returns JSON with 'detox_text' and 'explanation'.

Your task is to process the given text step by step by **calling these tools directly**. You are NOT allowed to assume, infer, or guess any results on your own. You MUST rely entirely on the tools to perform their respective actions.

Follow these steps strictly:

1. **Call LanguageDetector** to determine if the text is in English.
    - Do NOT attempt to detect language yourself.
    - If the result is False, assume the text is in Swahili and use SwahiliTranslator to translate it into English.

2. **Call SentimentAnalyzer** on the English text to analyze its sentiment.

3. **Call ToxicityAnalyzer** on the English text to analyze its toxicity.

4. If ToxicityAnalyzer identifies the text as toxic, **call Detoxifier** to produce a non-toxic version.

Do not skip or combine steps. Each step must involve calling the corresponding tool and using its output to inform the next step.

Do not produce a final answer until you have completed all steps.

Finally, return ONLY a flat JSON object where all values are strings, using exactly these keys:
- "original_sentence": the original input text
- "translated_sentence": the translated text if translation was performed, otherwise an empty string
- "sentiment_prediction": the predicted sentiment label (one of "positive", "negative", "mixed", "neutral")
- "sentiment_prediction_explanation": explanation for the sentiment prediction
- "toxicity_prediction": the predicted toxicity label (one of "toxic", "non-toxic")
- "toxicity_prediction_explanation": explanation for the toxicity prediction
- "detoxed_sentence": the detoxified text if detoxification was performed, otherwise an empty string

All intermediate thoughts and tool calls should use the tools exactly as defined. Do NOT rely on internal reasoning or external knowledge.

TEXT: {text}

"""
