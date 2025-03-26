import numpy as np
import pandas as pd
class HiddenMarkovModel():
    def __init__(self, transition_prob, emission_prob,begin_state = None):
        self.begin_state = begin_state
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob
        
    def viterbi(self):
        pass