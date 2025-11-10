import torch

def get_probability_dist(text: str, model, tokenizer, device) -> list[float]:
    """
    Gets the probability distribution over classes for the given text.
    
    :param text: Input text to classify.
    :param model: Trained text classification model.
    :param tokenizer: Tokenizer corresponding to the model.
    :param device: Device to run the computations on.
    :return: List of probabilities for each class.
    """
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs_dist = (
            torch.nn.functional.softmax(logits, dim=1).squeeze().detach().cpu().tolist()
        )
    return probs_dist
