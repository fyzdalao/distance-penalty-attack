import numpy as np





class my_class:
    def __init__(self, x=1):
        #np.random.seed(0)
        self.x = x

    def get_random_number(self):
        ret = np.random.random_sample()
        return ret
