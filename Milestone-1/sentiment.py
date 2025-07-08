from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from config import MODEL_ID, DEVICE_MAP, MAX_NEW_TOKENS
from langchain_huggingface import HuggingFacePipeline
from utils import run_chain
from prompt import SENTIMENT_PROMPT

def analyze_sentiment(llm, sents):
    """
    Analyze the sentiment of a list of sentences using a language model.

    Args:
        llm: The language model pipeline to use for inference.
        sents (List[str]): List of sentences to analyze.

    Returns:
        List[dict]: List of parsed responses, each containing:
            - 'sentiment_label': The sentiment classification ('positive', 'negative', 'mixed', or 'neutral').
            - 'explanation': A short explanation for the sentiment result.
    """
    # Define response schemas for the output parser
    sentiment_schema = ResponseSchema(
        name="sentiment_label",
        description="Result of sentiment analysis, return 'positive', 'negative', 'mixed' or 'neutral'.",
    )
    sentiment_explanation_schema = ResponseSchema(
        name="explanation",
        description="Short explanation for the result of sentiment analysis.",
    )
    sentiment_response_schemas = [sentiment_schema, sentiment_explanation_schema]
    sentiment_parser = StructuredOutputParser.from_response_schemas(sentiment_response_schemas)
    sentiment_format_instructions = sentiment_parser.get_format_instructions(only_json=True)

    # Use the provided prompt template for sentiment analysis
    sentiment_template = SENTIMENT_PROMPT

    # Run the chain over all sentences
    sentiment_responses = run_chain(
        sentiment_template,
        llm,
        sentiment_parser,
        sents,
        task_name="sentiment",
        format_instructions=sentiment_format_instructions
    )
    return sentiment_responses