import numpy as np
import pandas as pd
class HiddenMarkovModel():
    def __init__(self, transition_prob, emission_prob,begin_state = None):
        self.begin_state = begin_state
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob
        if self.begin_state is None:
            self.begin_state = pd.DataFrame(1, index=self.transition_prob.columns, columns=['begin'])
        
    def viterbi(self,sequence=None):
        """ tìm dãy label có xác suất cao nhất dựa trên dãy observation """
        # làm mịn bảng với laplace smoothing vì giải thuật này sử dụng nhân
        transition_table = self.transition_prob.copy()
         
        emission_table = self.emission_prob.copy()
        laplace_constant = 1

        num_tags = transition_table.shape[1]  # Số lượng tags
        num_words = emission_table.shape[0]   # Số lượng từ khác nhau
        transition_table += laplace_constant
        emission_table += laplace_constant
        transition_table = transition_table.div(transition_table.sum(axis=1), axis=0)
        emission_table = emission_table.div(emission_table.sum(axis=0) )
        
        # tiến hành viterbi
        state =[]
        prob = 1
        label = []
        for word_idx in range(len(sequence)):
            compare_list = []
            compare_label = []
            for label_idx in range(num_tags):
                if word_idx == 0:
                    compare_list.append(self.begin_state.loc[self.begin_state.index[label_idx],'begin']  * (emission_table.loc[sequence[word_idx],self.begin_state.index[label_idx]]) )
                else:
                    # Ở đây không nhân thêm với prob vì bản chất prob là hằng số , không quan trong trong việc xét ở mỗi quan sát , thêm nữa đặc trưng bài toán này xác suất trả ra rất thấp, nhân cộng dồn khiến xác suất bị tràn số
                    compare_list.append((transition_table.loc[label[word_idx-1],transition_table.columns[label_idx]]) * (emission_table.loc[sequence[word_idx],self.begin_state.index[label_idx]]) )
                
                compare_label.append(emission_table.columns[label_idx])

            max_prob_idx = np.argmax(compare_list)  
            
            max_prob = compare_list[max_prob_idx]  
            max_label = compare_label[max_prob_idx]  
            prob = prob * max_prob
            label.append(max_label)
        return label
                
    