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