from tqdm import tqdm
from langchain_core.exceptions import OutputParserException
from langchain.prompts import PromptTemplate

def safe_invoke(chain, text, parser, sentence_id, task_name="task", max_retries=3):
    response = None
    for attempt in range(1, max_retries + 1):
        try:
            response = chain.invoke({"text": text})
            return parser.parse(response)
        except OutputParserException as e:
            print(f"[{task_name}] Retry {attempt}/{max_retries} for sentence {sentence_id}")
            print(f"   OutputParserException: {e}")
            if attempt == max_retries:
                print(f"[{task_name}] Max retries reached for sentence {sentence_id}. Marking as failure.")
                return {
                    f"{task_name.lower()}_label": "unknown",
                    "explanation": "LLM failure"
                }

def run_chain_single(prompt_template, llm, parser, text, task_name, format_instructions):
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["text"],
        partial_variables={"format_instructions": format_instructions}
    )
    chain = prompt | llm
    return safe_invoke(chain, text, parser, sentence_id=0, task_name=task_name)
