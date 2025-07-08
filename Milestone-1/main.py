import argparse
from pipeline import run_analysis_pipeline, save_final_result
from config import SENTIMENT_INPUT, TOXICITY_INPUT, SENTIMENT_RESULT, TOXICITY_RESULT, MODEL_ID, DEVICE_MAP, MAX_NEW_TOKENS

def main():
    """
    Command-line interface for running the sentiment and toxicity analysis pipeline.

    Parses command-line arguments for input/output paths and model configuration,
    runs the analysis pipeline, and saves the results to a CSV file.
    """
    parser = argparse.ArgumentParser(description="Run sentiment and toxicity analysis pipeline.")
    parser.add_argument("--input", type=str, default=SENTIMENT_INPUT, help="Input CSV file path")
    parser.add_argument("--output", type=str, default=SENTIMENT_RESULT, help="Output CSV file path")
    parser.add_argument("--model_id", type=str, default=MODEL_ID, help="Model ID")
    parser.add_argument("--device_map", type=str, default=DEVICE_MAP, help="Device map")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS, help="Max new tokens")
    args = parser.parse_args()

    final_result = run_analysis_pipeline(
        args.input, 
        args.model_id, 
        args.device_map, 
        args.max_new_tokens
    )
    # Use the correct save function that handles list-to-DataFrame conversion
    save_final_result(final_result, args.output)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()