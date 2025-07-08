from tqdm import tqdm
from langchain_core.exceptions import OutputParserException
import pandas as pd
import os
from langchain.prompts import PromptTemplate

def load_df_and_sents(data_path):
    """
    Load a CSV file and extract the sentences column.

    Args:
        data_path (str): Path to the CSV file. The file must contain a 'sentence' column.

    Returns:
        tuple: (DataFrame, List[str]) - The loaded DataFrame and a list of sentences.
    """
    df = pd.read_csv(data_path)
    sentences = list(df.sentence)
    return df, sentences

def safe_invoke(chain, text, parser, sentence_id, task_name="sentiment", max_retries=3):
    """
    Invokes a LangChain chain with retries on OutputParserException.

    If all retries fail, returns a fallback result with 'unknown' label and explanation.

    Args:
        chain: LangChain chain object.
        text (str): The input text to process.
        parser: LangChain output parser.
        sentence_id (int): Index of the current sentence (for logging).
        task_name (str): Descriptive name for debug output (e.g., "sentiment", "toxicity").
        max_retries (int): Number of retry attempts.

    Returns:
        dict: Parsed response or fallback dict on failure.
    """
    response = None  # Ensure response is defined for exception handling
    for attempt in range(1, max_retries + 1):
        try:
            response = chain.invoke({"text": text})
            parsed = parser.parse(response)
            return parsed
        except OutputParserException as e:
            print(f"[{task_name}] Retry {attempt}/{max_retries} for sentence {sentence_id}")
            print(f"   OutputParserException: {e}")
            print(f"   Original Invalid Output: {response}")
            if attempt == max_retries:
                print(f"[{task_name}] Max retries reached for sentence {sentence_id}. Marking as failure.")
                return {
                    f"{task_name.lower()}_label": "unknown",
                    "explanation": "LLM failure"
                }

def run_chain(prompt_template, llm, parser, sents, task_name, format_instructions):
    """
    Runs a LangChain pipeline over a list of sentences.

    Args:
        prompt_template (str): The prompt template string.
        llm: The language model to use.
        parser: The output parser for the model's responses.
        sents (List[str]): List of sentences to process.
        task_name (str): Name of the task (for logging).
        format_instructions (str): Instructions for output formatting.

    Returns:
        List[dict]: List of parsed responses for each sentence.
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["text"],
        partial_variables={"format_instructions": format_instructions}
    )

    chain = prompt | llm
    results = []

    for idx, text in enumerate(tqdm(sents)):
        parsed_response = safe_invoke(
            chain, text, parser, sentence_id=idx, task_name=task_name
        )
        results.append(parsed_response)

    return results

def save_results(df, out_path):
    """
    Save a DataFrame to a CSV file, creating directories as needed.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        out_path (str): The output file path.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)