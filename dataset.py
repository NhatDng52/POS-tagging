import pandas as pd
from datasets import load_dataset
import numpy as np
# Tải dữ liệu từ Hugging Face
dataset = load_dataset("batterydata/pos_tagging")

# Chuyển tập train thành DataFrame
df_train = pd.DataFrame(dataset["train"])
df_test = pd.DataFrame(dataset["test"])

print(df_train.head())

