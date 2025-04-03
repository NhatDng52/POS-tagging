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
if __name__ == "__main__":
    pos = POS()
    dataset = POSDataset()
    pos.train(dataset.train)
    pos.test(dataset.test)
    