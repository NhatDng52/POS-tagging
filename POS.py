import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from HMM import HiddenMarkovModel
from dataset import POSDataset
from RNN import CustomRNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
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
    """ Class chính để thực hiện POS Tagging, gọi RNN model và viterbi """
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn = CustomRNN(input_dim, hidden_dim, output_dim)
    
    def train(self, train_data,label):
        """ Hàm train RNN model """
        # Chuyển đổi dữ liệu thành định dạng phù hợp với RNN
        emission_tokenizer = Tokenizer(oov_token="UNK")  # Định nghĩa token đặc biệt
        emission_tokenizer.fit_on_texts(train_data)
        X = emission_tokenizer.texts_to_sequences(train_data)
        X = pad_sequences(X, padding='post')
        X = np.expand_dims(X,axis=-1)
        pos_tokenizer = Tokenizer(oov_token="UNK")  # Định nghĩa token đặc biệt
        pos_tokenizer.fit_on_texts(label)
        Y = pos_tokenizer.texts_to_sequences(label)
        Y = pad_sequences(Y, padding='post')
        Y = np.expand_dims(Y,axis=-1)
        model = self.rnn(X,Y)
        
        
        
        
        
if __name__ == "__main__":
    pos = POS_RNN(input_dim=3, hidden_dim=4, output_dim=2)
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