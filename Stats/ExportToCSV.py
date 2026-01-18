import pandas as pd
df = pd.read_parquet("task_a_trial.parquet")
df.to_csv("task_a_trial.csv", index=False)