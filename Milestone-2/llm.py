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