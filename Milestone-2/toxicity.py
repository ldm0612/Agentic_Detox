from utils import run_chain_single
from prompt import TOXICITY_PROMPT
from langchain.output_parsers import ResponseSchema, StructuredOutputParser



def analyze_toxicity(llm, text):
    toxicity_schema = ResponseSchema(
        name="toxicity_label",
        description="Toxicity label: toxic or non-toxic."
    )
    toxicity_explanation_schema = ResponseSchema(
        name="explanation",
        description="Explanation of the toxicity label."
    )
    toxicity_parser = StructuredOutputParser.from_response_schemas([toxicity_schema, toxicity_explanation_schema])
    toxicity_format_instructions = toxicity_parser.get_format_instructions(only_json=True)
    
    result = run_chain_single(    
        TOXICITY_PROMPT, llm, toxicity_parser, text,
        task_name="toxicity", format_instructions=toxicity_format_instructions
    )
    return result