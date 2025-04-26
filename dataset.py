import pandas as pd
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
class POSDataset():
    """ Class này dùng để load dữ liệu từ Hugging Face """
    def __init__(self):
        dataset = load_dataset("batterydata/pos_tagging")
        self.train = pd.DataFrame(dataset["train"])
        self.test = pd.DataFrame(dataset["test"])
        self.df = self.concat_if_same_fields(self.train, self.test)

    def concat_if_same_fields(self, df1, df2):
        if list(df1.columns) != list(df2.columns):
            raise ValueError("Hai DataFrame không cùng cấu trúc cột.")
        return pd.concat([df1, df2], ignore_index=True)

    def cross_validation_split(self, k=None, n=None, shuffle=True):
        if k is None and n is None:
            raise ValueError("Phải chỉ định ít nhất một trong hai tham số k hoặc n.")
        elif k is not None and n is not None:
            raise ValueError("Chỉ được chỉ định một trong hai tham số k hoặc n.")

        df = self.df.copy()
        if shuffle:
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        total = len(df)

        if k is not None:
            if k > total:
                raise ValueError("Giá trị k vượt quá số lượng mẫu.")
            fold_size = total // k
            for i in range(k):
                val_start = i * fold_size
                val_end = val_start + fold_size if i != k - 1 else total
                val_df = df.iloc[val_start:val_end]
                train_df = df.drop(df.index[val_start:val_end])
                yield train_df, val_df
        else:  # chia theo n phần tử mỗi fold
            for i in range(0, total, n):
                val_df = df.iloc[i:i+n]
                train_df = df.drop(df.index[i:i+n])
                yield train_df, val_df

    def split_train_test_valid(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, shuffle=True):
        """Chia dữ liệu thành train, val, test theo tỉ lệ (không dùng sklearn)"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Tổng tỉ lệ phải bằng 1"

        df = self.df.copy()
        if shuffle:
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        total = len(df)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        return train_df, val_df, test_df

if __name__ == "__main__":
    dataset = POSDataset()

    print("Train shape:", dataset.train.to_numpy().shape)
    print("Test shape:", dataset.test.to_numpy().shape)

    # Combine labels from train and test
    train_labels = dataset.train.iloc[:, 1]  # Second column contains labels
    test_labels = dataset.test.iloc[:, 1]    # Second column contains labels

    # Flatten the lists of labels
    all_labels = [label for sublist in train_labels for label in sublist] + \
                 [label for sublist in test_labels for label in sublist]

    # Count the frequency of each label
    label_counts = Counter(all_labels)

    # Normalize to probabilities
    total_labels = sum(label_counts.values())
    label_probs = {label: count / total_labels for label, count in label_counts.items()}
    # Sort labels by frequency for better visualization
    sorted_labels = sorted(label_probs.keys(), key=lambda x: label_probs[x], reverse=True)
    sorted_probs = [label_probs[label] for label in sorted_labels]

    # Plot the probability distribution
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_labels, sorted_probs, color='skyblue')
    plt.xlabel('Labels')
    plt.ylabel('Probability')
    plt.title('Probability Distribution of POS Tags')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()