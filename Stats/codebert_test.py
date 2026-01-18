import os
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
)
from codebert_training import CodeDataset, compute_metrics


def main():
    model_dir = "./codebert-finetuned"
    test_path = "task_a_test_set_sample.parquet"

    # Check paths
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # Device selection: CUDA → MPS (Apple Silicon) → CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model.to(device)
    print(f"Using device: {device}")

    # Load test data
    test_df = pd.read_parquet(test_path)
    # Optional: test_df = test_df.head(50)

    test_dataset = CodeDataset(test_df, tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Evaluate
    metrics = trainer.evaluate(eval_dataset=test_dataset)

    print("\n===== Test Metrics =====")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
