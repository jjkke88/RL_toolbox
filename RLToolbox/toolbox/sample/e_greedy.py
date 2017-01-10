import random
class EGreedy(object):
    def __init__(self):
        self.e_greedy = 1

    def get_sample_type(self, e_greedy):
        if random.random() < e_greedy:
            return "RANDOM"
        else:
            return "POLICY"