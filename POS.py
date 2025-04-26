import os
import numpy as np
import pandas as pd
import torch
from collections import Counter

from HMM import HiddenMarkovModel
from dataset import POSDataset
from RNN import CustomRNN
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

def convert_sentences_to_embeddings(sentences, word2vec, embedding_dim=100):
    embedded_sentences = []

    for sentence in sentences:
        embedded_words = []
        for word in sentence:
            if word in word2vec:
                embedded_words.append(word2vec[word])
            else:
                embedded_words.append(np.zeros(embedding_dim))
        embedded_sentences.append(torch.tensor(np.array(embedded_words), dtype=torch.float32))

    return pad_sequences_torch(embedded_sentences)

def pad_sequences_torch(sequences, padding_value=0.0):
    max_len = max(seq.size(0) for seq in sequences)
    embedding_dim = sequences[0].size(1)
    padded = torch.full((len(sequences), max_len, embedding_dim), padding_value)
    for i, seq in enumerate(sequences):
        padded[i, :seq.size(0)] = seq
    return padded

def build_label_vocab(labels):
    counter = Counter()
    for label_seq in labels:
        counter.update(label_seq)
    idx_to_label = ["<unk>"] + list(counter.keys())
    label_to_idx = {label: idx for idx, label in enumerate(idx_to_label)}
    return label_to_idx

def to_categorical_torch(sequences, num_classes):
    batch_size = len(sequences)
    max_len = max(len(seq) for seq in sequences)
    tensor = torch.zeros((batch_size, max_len, num_classes))
    for i, seq in enumerate(sequences):
        for j, label in enumerate(seq):
            tensor[i][j][label] = 1
    return tensor

class POS_HMM():
    def train(self, data, label):
        # data, label đều là list các list
        assert len(data) == len(label), "Số lượng câu và nhãn không khớp."

        flatten_word = [word for sentence in data for word in sentence]
        flatten_label = [tag for tag_seq in label for tag in tag_seq]

        self.states = np.unique(flatten_label)
        self.emission = np.unique(flatten_word)

        transition_table = pd.DataFrame(0, index=self.states, columns=self.states)
        emission_table = pd.DataFrame(0, index=self.emission, columns=self.states)

        for tokens, tags in zip(data, label):
            for i in range(len(tokens)):
                emission_table.loc[tokens[i], tags[i]] += 1
                if i < len(tokens) - 1:
                    transition_table.loc[tags[i], tags[i+1]] += 1
        # print("Transition table shape:", transition_table.shape)
        # print("Emission table shape:", emission_table.shape)
        # print(emission_table)
        self.model = HiddenMarkovModel(transition_table, emission_table)
        self.model.laplace_smoothing()
        # _ = self.model.viterbi(data[0])  # warm up

    def test(self, data, label):
        assert len(data) == len(label), "Số lượng câu và nhãn không khớp."

        correct = 0 
        total = 0

        for tokens, tags in zip(data, label):
            out = self.model.viterbi(tokens)
            seq_correct, seq_total = self.__compare_string__(out, tags)
            correct += seq_correct
            total += seq_total

        acc = correct / total
        print("Accuracy:", acc)
        return acc

    def __compare_string__(self, list1, list2):
        if len(list1) != len(list2):
            raise ValueError("Hai list có độ dài khác nhau")
        same = sum(1 for a, b in zip(list1, list2) if a == b)
        return same, len(list1)

class POS_RNN():
    def __init__(self, input_dim, hidden_dim, output_dim, embedding_dim=100):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.rnn = CustomRNN(embedding_dim, hidden_dim, output_dim)

    def train(self, train_data, label):
        glove_file = 'glove.6B/glove.6B.100d.txt'
        word2vec_file = 'glove.6B/glove.6B.100d.word2vec.txt'
        if not os.path.exists(word2vec_file):
            glove2word2vec(glove_file, word2vec_file)

        self.word2vec = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)

        train_data = [s if isinstance(s, list) else s.split() for s in train_data]
        label = [s if isinstance(s, list) else s.split() for s in label]

        X = convert_sentences_to_embeddings(train_data, self.word2vec, embedding_dim=self.embedding_dim)

        self.label_vocab = build_label_vocab(label)
        self.idx_to_label = list(self.label_vocab.keys())
        Y_seq = [[self.label_vocab.get(token, 0) for token in seq] for seq in label]

        self.num_labels = len(self.label_vocab)
        Y_onehot = to_categorical_torch(Y_seq, self.num_labels)

        self.rnn.train_model(X, Y_onehot)

    def test(self, test_data, test_labels):
        test_data = [s if isinstance(s, list) else s.split() for s in test_data]
        test_labels = [s if isinstance(s, list) else s.split() for s in test_labels]

        X = convert_sentences_to_embeddings(test_data, self.word2vec, embedding_dim=self.embedding_dim)

        device = next(self.rnn.parameters()).device  # Lấy device của model
        X_tensor = X.clone().detach().to(device)

        Y_pred = self.rnn(X_tensor).detach().cpu().numpy()
        
        total = 0
        correct = 0

        for i in range(len(test_data)):
            pred_classes = np.argmax(Y_pred[i], axis=-1)
            true_classes = [self.label_vocab.get(token, 0) for token in test_labels[i]]

            for pred, true in zip(pred_classes, true_classes):
                if pred == true:
                    correct += 1
                total += 1

        print("Accuracy:", correct / total)
        return correct / total

if __name__ == "__main__":
    dataset = POSDataset()
    k = 2
    total_accuracy_rnn = 0
    total_accuracy_hmm = 0
    fold = 1

    for train_df, val_df in dataset.cross_validation_split(k=k):
        # print(f"\n===== Fold {fold} =====")

        x_train = [sentence for sentence in np.array(train_df)[:, 0]]
        y_train = [sentence for sentence in np.array(train_df)[:, 1]]
        x_val = [sentence for sentence in np.array(val_df)[:, 0]]
        y_val = [sentence for sentence in np.array(val_df)[:, 1]]

        # === RNN ===
        print(">> Training RNN")
        pos_rnn = POS_RNN(input_dim=1, hidden_dim=4, output_dim=50)
        pos_rnn.train(x_train, y_train)
        acc_rnn = pos_rnn.test(x_val, y_val)
        total_accuracy_rnn += acc_rnn

        # === HMM ===
        print(">> Training HMM")
        train_data = list(zip(x_train, y_train))
        val_data = list(zip(x_val, y_val))
        pos_hmm = POS_HMM()
        pos_hmm.train(x_train,y_train)
        acc_hmm = pos_hmm.test(x_val, y_val)
        total_accuracy_hmm += acc_hmm

        fold += 1

    print("\n====== Tổng kết ======")
    print("RNN average accuracy:", total_accuracy_rnn / k)
    print("HMM average accuracy:", total_accuracy_hmm / k)