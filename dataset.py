import pandas as pd
from datasets import load_dataset
import numpy as np

# Tải dữ liệu từ Hugging Face
# dataset = load_dataset("batterydata/pos_tagging")

# Chuyển tập train thành DataFrame
# df_train = pd.read_csv("train_data.csv")
# df_test = pd.read_csv("test_data.csv")

# print(dataset.keys())

class POSDataset():
    """ Class này dùng để load dữ liệu từ Hugging Face """
    def __init__(self):
        dataset = load_dataset("batterydata/pos_tagging")
        self.train = pd.DataFrame(dataset["train"]).head(2)
        self.test = pd.DataFrame(dataset["test"]).head(2)
