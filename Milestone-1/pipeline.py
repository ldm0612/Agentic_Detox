from utils import load_df_and_sents
from langchain_huggingface import HuggingFacePipeline
import pandas as pd
from config import MODEL_ID, DEVICE_MAP, MAX_NEW_TOKENS, SENTIMENT_INPUT, TOXICITY_INPUT, SENTIMENT_RESULT, TOXICITY_RESULT
from sentiment import analyze_sentiment
from toxicity import analyze_toxicity

def get_llm(model_id, device_map=DEVICE_MAP, max_new_tokens=MAX_NEW_TOKENS):
    """
    Initialize a HuggingFace language model pipeline for text generation.

    Args:
        model_id (str): The model identifier.
        device_map (str or dict): Device mapping for model inference.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        HuggingFacePipeline: Configured language model pipeline.
    """
    return HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        device_map=device_map,
        pipeline_kwargs={
            "return_full_text": False,
            "max_new_tokens": max_new_tokens
        }
    )

def run_analysis_pipeline(
    data_path=SENTIMENT_INPUT,
    model_id=MODEL_ID,
    device_map=DEVICE_MAP,
    max_new_tokens=MAX_NEW_TOKENS
):
    """
    Run sentiment and toxicity analysis on a dataset.

    Args:
        data_path (str): Path to the input CSV file containing sentences.
        model_id (str): The model identifier.
        device_map (str or dict): Device mapping for model inference.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        List[dict]: List of results, each containing sentence, sentiment, and toxicity predictions and explanations.
    """
    df, sents = load_df_and_sents(data_path)
    llm = get_llm(model_id, device_map, max_new_tokens)

    # Sentiment Analysis
    print("Performing Sentiment Analysis")
    sentiment_responses = analyze_sentiment(llm, sents)

    # Toxicity Analysis
    print("Performing Toxicity Analysis")
    toxicity_responses = analyze_toxicity(llm, sents)

    print("Integrating Results")
    final_result = []
    for i, sent in enumerate(sents):
        sentiment_result = sentiment_responses[i]
        toxicity_result = toxicity_responses[i]
        sent_result = {
            "id": i,
            "sentence": sent,
            "sentiment_prediction": sentiment_result['sentiment_label'],
            "sentiment_prediction_explanation": sentiment_result['explanation'],
            "toxicity_prediction": toxicity_result['toxicity_label'],
            "toxicity_prediction_explanation": toxicity_result['explanation'],
        }
        final_result.append(sent_result)

    print("Result Integrated Successfully!")
    return final_result

def save_final_result(final_result, out_path):
    """
    Save the final analysis results to a CSV file.

    Args:
        final_result (List[dict]): List of result dictionaries.
        out_path (str): Output file path.
    """
    df = pd.DataFrame(final_result)
    df.to_csv(out_path, index=False)