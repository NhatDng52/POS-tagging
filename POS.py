import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import tensorflow as tf
from HMM import HiddenMarkovModel
from dataset import POSDataset
from RNN import CustomRNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU devices:", gpus)
else:
    print("No GPU available.")
def convert_sentences_to_embeddings(sentences, word2vec, embedding_dim=100):
    embedded_sentences = []

    for sentence in sentences:
        embedded_words = []
        for word in sentence:
            if word in word2vec:
                embedded_words.append(word2vec[word])
            else:
                embedded_words.append(np.zeros(embedding_dim))  # từ không có trong word2vec
        embedded_sentences.append(embedded_words)

    return pad_sequences(embedded_sentences, padding='post', dtype='float32')
class POS_HMM() : 
    """ Class chính để thực hiện POS Tagging, gọi HMM model và viterbi """
    def train(self,train_data):
        """ Hàm train HMM model """
        train_data = np.array(train_data)
        #train_data = train_data[0]
        train_data[0][0][1]
        flatten_label = []
        for i in range(len(train_data[:,1])):
            flatten_label+= train_data[:,1][i]  
        flatten_word = []
        for i in range(len(train_data[:,0])):
            flatten_word+= train_data[:,0][i] 
        self.states = np.unique(flatten_label)
        transition_table = pd.DataFrame(0, index=self.states, columns=self.states)
        self.emission = np.unique(flatten_word)
        emission_table = pd.DataFrame(0, index=self.emission, columns=self.states)
        
        for samples in train_data:
            for i in range(len(samples[0])):
                emission_table.loc[samples[0][i], samples[1][i]] += 1
                if i == len(samples[0])-1:
                    break
                
                transition_table.loc[samples[1][i], samples[1][i+1]] += 1
        # print(emission_table)
        self.model = HiddenMarkovModel(transition_table, emission_table)
        self.model.laplace_smoothing()
        # print(train_data[0][0])
        self.model.viterbi(train_data[0][0])
    def test(self,test_data):
        """ Hàm đánh giá model """
        correct = 0 
        total = 0
        test_data = np.array(test_data) if type(test_data) == pd.DataFrame else test_data
        for sequence in test_data:
            out = self.model.viterbi(sequence[0])
            seq_correct,seq_total = self.__compare_string__(out,sequence[1])
            correct += seq_correct
            total += seq_total
        print("Acuracy: ", correct/total) 
            
        
    def create_test(self):
        """ Hàm tự tạo test dữ liệu"""
        pass
    def __compare_string__(self,list1,list2):
        """ hàm so sánh 2 list nhãn có độ dà bằng nhau"""
        same = 0
        total = 0
        if len(list1) != len(list2):
            raise ValueError("Hai list có độ dài khác nhau")
        for i in range(len(list1)):
            total +=1
            if list1[i] == list2[i]:
                same +=1
        return (same,total)
# if __name__ == "__main__":
#     pos = POS()
#     dataset = POSDataset()
#     pos.train(dataset.train)
#     pos.test(dataset.test)
    
    

class POS_RNN():
    def __init__(self, input_dim, hidden_dim, output_dim, embedding_dim=100):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.rnn = CustomRNN(embedding_dim, hidden_dim, output_dim)

    def train(self, train_data, label):
        # Nếu chưa có file word2vec, convert từ glove trước
        glove_file = 'glove.6B/glove.6B.100d.txt'
        word2vec_file = 'glove.6B/glove.6B.100d.word2vec.txt'
        if not os.path.exists(word2vec_file):
            glove2word2vec(glove_file, word2vec_file)

        # Load vectors
        self.word2vec = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)

        # Đảm bảo train_data và label là list các list từ
        train_data = [s if isinstance(s, list) else s.split() for s in train_data]
        label = [s if isinstance(s, list) else s.split() for s in label]

        # Convert X
        X = convert_sentences_to_embeddings(train_data, self.word2vec, embedding_dim=self.embedding_dim)

        # Tokenize nhãn và one-hot
        self.label_tokenizer = Tokenizer(oov_token="UNK")
        self.label_tokenizer.fit_on_texts(label)
        Y_seq = self.label_tokenizer.texts_to_sequences(label)
        Y_seq = pad_sequences(Y_seq, padding='post')

        self.num_labels = len(self.label_tokenizer.word_index) + 1  # +1 vì bắt đầu từ 1
        Y_onehot = np.array([to_categorical(seq, num_classes=self.num_labels) for seq in Y_seq])
        print("Y_onehot shape:", Y_onehot.shape)  # shape: (batch, seq_len, num_labels)
        print("X shape:", X.shape)
        # Train
        self.rnn.train_model(X, Y_onehot)

    def test(self, test_data, test_labels):
        test_data = [s if isinstance(s, list) else s.split() for s in test_data]
        test_labels = [s if isinstance(s, list) else s.split() for s in test_labels]

        X = convert_sentences_to_embeddings(test_data, self.word2vec, embedding_dim=self.embedding_dim)
        Y_pred = self.rnn(X).numpy()  # shape: (batch, seq_len, num_labels)

        total = 0
        correct = 0

        for i in range(len(test_data)):
            pred_classes = np.argmax(Y_pred[i], axis=-1)
            true_classes = self.label_tokenizer.texts_to_sequences([test_labels[i]])[0]

            for pred, true in zip(pred_classes, true_classes):
                if pred == true:
                    correct += 1
                total += 1

        print("Accuracy:", correct / total)
if __name__ == "__main__":
    pos = POS_RNN(input_dim=1, hidden_dim=4, output_dim=50)
    dataset = POSDataset()
    x = []
    for sentence in np.array(dataset.train)[:, 0]:
        x.append(sentence)
    y = []
    for sentence in np.array(dataset.train)[:, 1]:
        y.append(sentence)
    # print(np.array(dataset.train)[0][0])
    # print(x)
    pos.train(x,y)
    pos.test(x,y)