# -*- coding: utf-8 -*
import tiktoken


# Get pricing details from OpenAI's pricing page
PRICING = {
    "gpt-4o": {"input": 2.5 * 1e-6, "output": 10.0 * 1e-6},
    # Add more models and their pricing as needed
}


def estimate_cost_one(model: str, prompt: str, response_tokens: int) -> float:
    """Estimate the cost of an OpenAI API call.

    Args:
        model (str): The name of the model.
        prompt (str): The input prompt given to the model.
        response_tokens (int): Estimate of how many output tokens the model will return.

    Returns:
        A float representing the estimated cost of the API call in US dollars.

    Raises:
        ValueError: If no pricing data is configured for the model.
    """
    encoding = tiktoken.encoding_for_model(model)
    prompt_tokens = len(encoding.encode(prompt))

    # Get pricing details from OpenAI's pricing page
    model_pricing = PRICING.get(model, {})
    if not model_pricing:
        raise ValueError(f"Pricing for model '{model}' not found.")

    prompt_cost = prompt_tokens * model_pricing["input"]
    response_cost = response_tokens * model_pricing["output"]
    return prompt_cost + response_cost


def estimate_cost(model: str, prompts: list[str], response_tokens: int = 10) -> float:
    """Estimate the cost of an OpenAI API call.

    Args:
        model (str): The name of the model.
        prompt (list[str]): The input prompts given to the model.
        response_tokens (int): Estimate of the mean number of output tokens per prompt.

    Returns:
        A float representing the estimated cost of the API calls in US dollars.
    """
    return sum(map(lambda p: estimate_cost_one(model, p, response_tokens), prompts))
