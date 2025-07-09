from utils import run_chain_single
from prompt import DETOX_PROMPT
from langchain.output_parsers import ResponseSchema, StructuredOutputParser


def detoxify_text(llm, text):
    detox_schema = ResponseSchema(
        name="detox_text",
        description="The rewritten, detoxified version of the text."
    )
    detox_explanation_schema = ResponseSchema(
        name="explanation",
        description="Short explanation of how the text was modified."
    )
    detox_parser = StructuredOutputParser.from_response_schemas([detox_schema, detox_explanation_schema])
    detox_format_instructions = detox_parser.get_format_instructions(only_json=True)

    result = run_chain_single(
        DETOX_PROMPT, llm, detox_parser, text,
        task_name="detoxification", format_instructions=detox_format_instructions
    )
    
    return result
