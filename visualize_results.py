import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from captum.attr import LayerIntegratedGradients, visualization


def plot_radar(probs: list[float], labels: list[str]) -> None:
    """
    Plots a radar chart with all morals and corresponding probabilities.

    :param probs: List of probabilities for each moral class.
    :param labels: List of moral class labels.
    """
    values = probs.copy()
    num_val = len(values)

    angles = np.linspace(0, 2 * np.pi, num_val, endpoint=False).tolist()

    values += values[:1]
    angles += angles[:1]

    _, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(angles, values, color="red", linewidth=1)
    ax.fill(angles, values, color="red", alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)

    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment("center")
        elif 0 < angle < np.pi:
            label.set_horizontalalignment("left")
        else:
            label.set_horizontalalignment("right")

    ax.set_ylim(0, round(max(values), 1) + 0.1)
    ax.set_rlabel_position(180 / num_val)

    ax.set_title("Distribution of the answer", va="bottom")
    plt.show()


def visualize_token_importance(
    text: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    id2label: dict,
    device: torch.device,
    max_prob_thresh: float = 0.1,
    max_classes=np.inf,
) -> None:
    """
    Visualizes token importance using Integrated Gradients.

    :param text: Input text to analyze.
    :param model: Trained text classification model.
    :param tokenizer: Tokenizer corresponding to the model.
    :param id2label: Mapping from label IDs to label names.
    :param device: Device to run the computations on.
    :param max_prob_thresh: Maximum probability threshold to consider a class for visualization.
    :param max_classes: Maximum number of classes to visualize.
    """

    def _forward_func(inputs, position=0):
        outputs = model(inputs, attention_mask=torch.ones_like(inputs))
        return outputs[position]

    with torch.no_grad():
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        ).to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        probs_dist = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    prediction_tuples = [(id2label[i], p) for i, p in enumerate(probs_dist)]
    sorted_prediction_tuples = sorted(
        prediction_tuples, key=lambda x: x[1], reverse=True
    )
    pred_class, pred_prob = sorted_prediction_tuples[0]

    layer = getattr(model, "distilbert").embeddings
    lig = LayerIntegratedGradients(_forward_func, layer)

    inputs = torch.tensor(
        tokenizer.encode(text, add_special_tokens=False), device=device
    ).unsqueeze(0)
    tokens = tokenizer.convert_ids_to_tokens(inputs.detach().cpu().numpy()[0])
    sequence_len = inputs.shape[1]
    baseline = torch.tensor(
        [tokenizer.cls_token_id]
        + [tokenizer.pad_token_id] * (sequence_len - 2)
        + [tokenizer.sep_token_id],
        device=device,
    ).unsqueeze(0)

    vis_record_l = []
    for i, (attr_class, attr_score) in enumerate(sorted_prediction_tuples):
        if (attr_score > max_prob_thresh) and (i < max_classes):
            target = model.config.label2id[attr_class]

            with torch.no_grad():
                attributes, delta = lig.attribute(
                    inputs=inputs,
                    baselines=baseline,
                    target=target,
                    return_convergence_delta=True,
                )

            attr = attributes.sum(dim=2).squeeze(0)
            attr = attr / torch.norm(attr)
            attr = attr.cpu().detach().numpy()

            vis_record = visualization.VisualizationDataRecord(
                word_attributions=attr,
                pred_prob=pred_prob,
                pred_class=pred_class,
                true_class=pred_class,
                attr_class=attr_class,
                attr_score=attr_score,
                raw_input_ids=tokens,
                convergence_score=delta,
            )
            vis_record_l.append(vis_record)

    _ = visualization.visualize_text(vis_record_l)
