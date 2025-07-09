from utils import run_chain_single
from prompt import SENTIMENT_PROMPT
from langchain.output_parsers import ResponseSchema, StructuredOutputParser


def analyze_sentiment(llm, text):
    sentiment_schema = ResponseSchema(
        name="sentiment_label",
        description="Sentiment label: positive, negative, mixed, or neutral."
    )
    sentiment_explanation_schema = ResponseSchema(
        name="explanation",
        description="Explanation of the sentiment."
    )
    sentiment_parser = StructuredOutputParser.from_response_schemas([sentiment_schema, sentiment_explanation_schema])
    sentiment_format_instructions = sentiment_parser.get_format_instructions(only_json=True)
    
    result = run_chain_single(
        SENTIMENT_PROMPT, llm, sentiment_parser, text,
        task_name="sentiment", format_instructions=sentiment_format_instructions
    )
    return result
