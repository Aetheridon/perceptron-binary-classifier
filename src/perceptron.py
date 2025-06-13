import random

class Perceptron:
    def __init__(self, data):
        self.data = data
        self.weight_1, self.weight_2 = random.uniform(-1, 1), random.uniform(-1, 1)
        self.bias = random.uniform(-1, 1)

    def hypothesis(self, humidity, pressure):
        score = self.weight_1 * humidity + self.weight_2 * pressure + self.bias
        classifier = self.step(score)
        return classifier
    
    def train(self, data_split):
        training_data = self.data[:data_split]
        # More to come
    
    def step(self, score):
        return 1 if score >= 0 else 0