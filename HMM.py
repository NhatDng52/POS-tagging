import numpy as np
import pandas as pd
class HiddenMarkovModel():
    def __init__(self, transition_prob, emission_prob,begin_state = None):
        self.begin_state = begin_state
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob
        if self.begin_state is None:
            self.begin_state = pd.DataFrame(1, index=self.transition_prob.columns, columns=['begin'])
        self.num_tags = self.transition_prob.shape[1]  # Số lượng tags
        self.num_words = self.emission_prob.shape[0]   # Số lượng từ khác nhau
        self.unk = pd.DataFrame(0, columns=self.emission_prob.columns, index=['<unk>'])
        # print(self.transition_prob.shape)
        # print(self.emission_prob.shape)
        # print(self.begin_state.shape)
        # print(self.unk.shape)
        self.update_unk()
    def laplace_smoothing(self):
        laplace_constant = 1
        transition_table = self.transition_prob.copy()
        emission_table = self.emission_prob.copy()
        transition_table += laplace_constant
        emission_table += laplace_constant
        transition_table = transition_table.div(transition_table.sum(axis=1), axis=0)
        emission_table = emission_table.div(emission_table.sum(axis=0) )
        self.transition_prob = transition_table
        self.emission_prob = emission_table
        
    def viterbi(self,sequence=None):
        """ tìm dãy label có xác suất cao nhất dựa trên dãy observation """
        # làm mịn bảng với laplace smoothing vì giải thuật này sử dụng nhân
        
        # tiến hành viterbi
        state =[]
        prob = 1
        label = []
        for word_idx in range(len(sequence)):
            compare_list = []
            compare_label = []
            """ 
            xử lí dữ liệu chưa thấy :
            check xem có trong index không 
            nếu có code dưới , nếu không store dữ liệu sau khi gọi hàm  
        
            """
            if sequence[word_idx] not in self.emission_prob.index:
                sequence[word_idx]= '<unk>'
            for label_idx in range(self.num_tags):
                if word_idx == 0:
                    compare_list.append(self.begin_state.loc[self.begin_state.index[label_idx],'begin']  * (self.emission_prob.loc[sequence[word_idx],self.begin_state.index[label_idx]]) )
                else:
                    # Ở đây không nhân thêm với prob vì bản chất prob là hằng số , không quan trong trong việc xét ở mỗi quan sát , thêm nữa đặc trưng bài toán này xác suất trả ra rất thấp, nhân cộng dồn khiến xác suất bị tràn số
                    compare_list.append((self.transition_prob.loc[label[word_idx-1],self.transition_prob.columns[label_idx]]) * (self.emission_prob.loc[sequence[word_idx],self.begin_state.index[label_idx]]) )
                
                compare_label.append(self.emission_prob.columns[label_idx])

            max_prob_idx = np.argmax(compare_list)  
            
            max_prob = compare_list[max_prob_idx]  
            max_label = compare_label[max_prob_idx]  
            prob = prob * max_prob
            label.append(max_label)
        return label             
    def update_unk(self):
        """ Update the unk table based on the most probable label for each word in emission_prob, then concat it to the emission_prob """
        for word in self.emission_prob.index:
            most_probable_label = self.emission_prob.loc[word].idxmax()
            self.unk.loc['<unk>',most_probable_label ] += 1
        self.emission_prob = pd.concat([self.emission_prob, self.unk])

         