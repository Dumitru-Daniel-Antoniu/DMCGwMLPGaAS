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
    preds = logits.argmax(axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"macro_f1": macro_f1, "accuracy": acc}


def main():
    train_path = "task_a_trial.parquet"
    val_path = "task_a_validation_set.parquet"
    test_path = "task_a_test_set_sample.parquet"

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    train_df = train_df.head(1000)
    val_df = val_df.head(500)
    test_df = test_df.head(50)

    print("train rows:", len(train_df))
    print("val rows:", len(val_df))

    model_name = "microsoft/unixcoder-base"
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
    test_dataset = CodeDataset(test_df, tokenizer)

    training_args = TrainingArguments(
        output_dir="./unixcoder-results",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.train()

    # metrics = trainer.evaluate()
    # print("Final metrics:", metrics)

    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print("Final metrics:", test_metrics)

    trainer.save_model("./unixcoder-finetuned")
    tokenizer.save_pretrained("./unixcoder-finetuned")
    print("Model saved to ./unixcoder-finetuned")


if __name__ == "__main__":
    main()
