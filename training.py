import fire
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
import torch.nn as nn
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import EarlyStoppingCallback

TEXT_COL = "text"
ANNOT_COL = "unified_annotation"
LABEL_COL = "label"


def load_data(path: str) -> tuple[pd.DataFrame, dict, dict]:
    """
    Loads and processes the dataset from a CSV file.
    """
    df = pd.read_csv(path)[[TEXT_COL, ANNOT_COL]]
    df.rename(columns={ANNOT_COL: LABEL_COL}, inplace=True)
    id2label = dict(enumerate(df[LABEL_COL].unique()))
    label2id = {v: k for k, v in id2label.items()}
    df[LABEL_COL] = df[LABEL_COL].map(label2id)
    return df, id2label, label2id


def calculate_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    Calculates class weights to handle class imbalance.
    """
    labels = np.array(df[LABEL_COL])
    classes = np.array(df[LABEL_COL].unique())
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    return class_weights


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def split_dataset(df: pd.DataFrame, tokenizer) -> tuple[Dataset, Dataset]:
    """
    Splits the dataset into training and validation sets and tokenizes the text.
    """
    X = list(df[TEXT_COL])
    y = list(df[LABEL_COL])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    X_train_tokenized = tokenizer(
        X_train, padding=True, truncation=True, max_length=512
    )
    X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)
    train_dataset = Dataset(X_train_tokenized, y_train)
    val_dataset = Dataset(X_val_tokenized, y_val)
    return train_dataset, val_dataset


def prepare_compute_metrics(label_names: list[str]):
    def compute_metrics(p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)

        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        weighted_recall = recall_score(y_true=labels, y_pred=pred, average="weighted")
        weighted_precision = precision_score(
            y_true=labels, y_pred=pred, average="weighted"
        )
        weighted_f1 = f1_score(y_true=labels, y_pred=pred, average="weighted")
        metrics = {
            "accuracy": accuracy,
            "weighted_recall": weighted_recall,
            "weighted_precision": weighted_precision,
            "weighted_f1": weighted_f1,
        }

        recalls = recall_score(y_true=labels, y_pred=pred, average=None)
        precisions = precision_score(y_true=labels, y_pred=pred, average=None)
        f1_scores = f1_score(y_true=labels, y_pred=pred, average=None)
        for i, label in enumerate(label_names):
            if label not in ("Non-Moral", "Thin Morality"):
                metrics.update(
                    {
                        f"recall_{label}": recalls[i],
                        f"precision_{label}": precisions[i],
                        f"f1_score_{label}": f1_scores[i],
                    }
                )

        return metrics

    return compute_metrics


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, self.class_weights.size(0)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def train(dataset_path: str, model_name: str, output_path: str) -> None:
    """
    Trains and saves a text classification model.

    :param dataset_path: Path to the CSV dataset file.
    :param model_name: Pretrained model name or path.
    :param output_path: Path to directory to save the trained model.
    """

    df, id2label, label2id = load_data(dataset_path)
    labels = list(label2id.keys())
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
    )

    args = TrainingArguments(
        output_dir="output",
        eval_strategy="steps",
        eval_steps=500,
        # max_steps=100,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        seed=0,
        load_best_model_at_end=True,
    )

    train_dataset, val_dataset = split_dataset(df, tokenizer)
    class_weights = calculate_class_weights(df)

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=prepare_compute_metrics(labels),
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=class_weights,
    )

    trainer.train()
    trainer.save_model(output_path)


if __name__ == "__main__":
    fire.Fire(train)
