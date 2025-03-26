import numpy as np
import pandas as pd
from HMM import HiddenMarkovModel
from dataset import POSDataset
class POS() : 
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
        
        # self.states = [str(state) for state in self.states]
        transition_table = pd.DataFrame(0, index=self.states, columns=self.states)
        # print(transition_table.columns)
        self.emission = np.unique(flatten_word)
        # self.emission = [str(word) for word in self.emission]
        # print(self.emission[0])
        emission_table = pd.DataFrame(0, index=self.emission, columns=self.states)
        # print(emission_table.columns)
        # print(emission_table.index)
        # Tính transition_table và emission_table  
        # print("Rows (index):", emission_table.index.to_list())
        # print("Columns:", emission_table.columns.to_list()) 
        # print(emission_table.loc['Confidence','NN'])
        
        for samples in train_data:
            for i in range(len(samples[0])):
                emission_table.loc[samples[0][i], samples[1][i]] += 1
                if i == len(samples[0])-1:
                    break
                
                transition_table.loc[samples[1][i], samples[1][i+1]] += 1
        transition_table = transition_table.div(transition_table.sum(axis=0), axis=1)
        emission_table = emission_table.div(emission_table.sum(axis=0), axis=1)
        self.model = HiddenMarkovModel(transition_table, emission_table)
    def test(self,test_data):
        """ Hàm đánh giá model """
        print(self.model.viterbi(test_data))
    def create_test(self):
        """ Hàm tự tạo test dữ liệu"""
        pass

if __name__ == "__main__":
    pos = POS()
    dataset = POSDataset()
    pos.train(dataset.train)
    