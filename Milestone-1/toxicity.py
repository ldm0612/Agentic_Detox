from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from config import MODEL_ID, DEVICE_MAP, MAX_NEW_TOKENS
from langchain_huggingface import HuggingFacePipeline
from utils import run_chain
from prompt import TOXICITY_PROMPT

def analyze_toxicity(llm, sents):
    """
    Analyze the toxicity of a list of sentences using a language model.

    Args:
        llm: The language model pipeline to use for inference.
        sents (List[str]): List of sentences to analyze.

    Returns:
        List[dict]: List of parsed responses, each containing:
            - 'toxicity_label': The toxicity classification ('toxic' or 'non-toxic').
            - 'explanation': A short explanation for the toxicity result.
    """
    # Define response schemas for the output parser
    toxicity_schema = ResponseSchema(
        name="toxicity_label",
        description="Result of toxicity analysis, return 'toxic' or 'non-toxic'.",
    )
    toxicity_explanation_schema = ResponseSchema(
        name="explanation",
        description="Short explanation for the result of toxicity analysis.",
    )
    toxicity_response_schemas = [toxicity_schema, toxicity_explanation_schema]
    toxicity_parser = StructuredOutputParser.from_response_schemas(toxicity_response_schemas)
    toxicity_format_instructions = toxicity_parser.get_format_instructions(only_json=True)

    # Use the provided prompt template for toxicity analysis
    toxicity_template = TOXICITY_PROMPT

    # Run the chain over all sentences
    toxicity_responses = run_chain(
        toxicity_template,
        llm,
        toxicity_parser,
        sents,
        task_name="toxicity",
        format_instructions=toxicity_format_instructions
    )
    return toxicity_responses