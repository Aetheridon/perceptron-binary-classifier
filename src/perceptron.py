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
    
    def train(self, data_split, learning_rate=0.01, epochs=10):
        '''Responsible for caliberating the biases and weights'''
        training_data = self.data[:data_split]
        
        for i in range(epochs):
            print(f"Training: on Epoch {i}")
            for x, row in training_data.iterrows():
                x1 = row["humidity"]
                x2 = row["pressure"]
                r_true = row["rain"] # Whether rain happened or not, represented by 1 or 0 respectively.
                print(f"Training: Training with H: {x1} & P: {x2} with result {r_true}")

                prediction = self.hypothesis(x1, x2)
                s_value = self.step(prediction)
                error = r_true - prediction
                print(f"Training: Predicted with score {prediction} and a step value of {s_value}, error margin is {error}")

                # Update weights and biases based off the error margin
                self.weight_1 += learning_rate * error * x1
                self.weight_2 += learning_rate * error * x2
                self.bias += learning_rate * error # This variable represents how well our model fits the relationship between the data
                print(f"Training: Weight_1 updated: {self.weight_1}, Weight_2 updated: {self.weight_2}, Bias updated: {self.bias}")

    def step(self, score):
        return 1 if score >= 0 else 0