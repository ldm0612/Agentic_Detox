from language_id import LanguageDetector
from translation import SwahiliTranslator
from sentiment import analyze_sentiment
from toxicity import analyze_toxicity
from detox import detoxify_text
from config import MODEL_ID, DEVICE_MAP, MAX_NEW_TOKENS
from langchain.agents import initialize_agent, AgentType
from langchain.agents import Tool
from prompt import AGENT_PROMPT
from llm import get_llm

import json

llm = get_llm(MODEL_ID, device_map=DEVICE_MAP, max_new_tokens=MAX_NEW_TOKENS)
language_detector = LanguageDetector()
sw_translator = SwahiliTranslator()

def get_tools():
    language_detector_tool = Tool(
        name="LanguageDetector",
        func=language_detector.detect_eng,
        description="Detects if a text is English. Returns True if English, False otherwise."
    )

    swahili_translator_tool = Tool(
        name="SwahiliTranslator",
        func=sw_translator.translate,
        description="Translates Swahili text into English."
    )

    sentiment_tool = Tool(
        name="SentimentAnalyzer",
        func=lambda text: analyze_sentiment(llm, text),
        description="Analyzes the sentiment of English text. Returns JSON with 'sentiment_label' and 'explanation'."
    )

    toxicity_tool = Tool(
        name="ToxicityAnalyzer",
        func=lambda text: analyze_toxicity(llm, text),
        description="Analyzes the toxicity of English text. Returns JSON with 'toxicity_label' and 'explanation'."
    )

    detox_tool = Tool(
        name="Detoxifier",
        func=lambda text: detoxify_text(llm, text),
        description="Rewrites toxic text to make it non-toxic while preserving meaning. Returns JSON with 'detox_text' and 'explanation'."
    )

    tools = [
        language_detector_tool,
        swahili_translator_tool,
        sentiment_tool,
        toxicity_tool,
        detox_tool
    ]
    return tools

def get_agent(tools, llm):
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    return agent

def process_texts_agentic(texts):
    """
    Process a list of texts using the agent to reason about the workflow:
    - Detect language and translate if necessary
    - Analyze sentiment
    - Analyze toxicity
    - Detoxify if toxic
    Returns a list of dictionaries with the final results.
    """
    results = []
    tools = get_tools()
    agent = get_agent(tools, llm)

    for idx, text in enumerate(texts):
        print(f"\nProcessing sentence {idx}: {text}\n")

        # Compose the agent instruction
        query = AGENT_PROMPT.format(text=text)

        # Let the agent reason and execute tool calls
        try:
            output = agent.run(query)
            # Safely parse JSON string from agent response
            parsed_output = json.loads(output)
        except Exception as e:
            print(f"Error processing sentence {idx}: {e}")
            parsed_output = {
                "original_sentence": text,
                "translated_sentence": "",
                "sentiment_prediction": "unknown",
                "sentiment_prediction_explanation": "Agent failed to process the text.",
                "toxicity_prediction": "unknown",
                "toxicity_prediction_explanation": "Agent failed to process the text.",
                "detoxed_sentence": ""
            }

        # Add sentence ID and save result
        parsed_output["id"] = idx
        results.append(parsed_output)

    return results