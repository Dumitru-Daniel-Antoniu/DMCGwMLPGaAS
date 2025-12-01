import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import f1_score, accuracy_score


class CodeDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        code = str(row["code"])
        label = int(row["label"])

        enc = self.tokenizer(
            code,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Sometimes predictions come as (logits,) from the model
    if isinstance(logits, tuple):
        logits = logits[0]

    preds = logits.argmax(axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"macro_f1": macro_f1, "accuracy": acc}


def main():
    train_path = "task_a_trial.parquet"
    val_path = "task_a_validation_set.parquet"

    # Load data
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    # For quick experiments – you can remove these .head() later
    train_df = train_df.head(1000)
    val_df = val_df.head(500)

    print("train rows:", len(train_df))
    print("val rows:", len(val_df))

    # CodeT5 model
    model_name = "Salesforce/codet5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("Using device:", device)

    train_dataset = CodeDataset(train_df, tokenizer)
    val_dataset = CodeDataset(val_df, tokenizer)

    # IMPORTANT: only use arguments supported by your transformers version
    training_args = TrainingArguments(
        output_dir="./codet5-results",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        learning_rate=2e-5,   # try 5e-5 later if you want
        weight_decay=0.01,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()

    # Explicit validation after training
    val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    print("\n===== Validation Metrics =====")
    for k, v in val_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # Save model + tokenizer
    save_dir = "./codet5-finetuned"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\nModel saved to {save_dir}")


if __name__ == "__main__":
    main()