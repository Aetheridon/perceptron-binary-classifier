from perceptron import Perceptron
import pandas as pd

df = pd.read_csv("mock_weather_data.csv")
df.dropna(inplace=True)
model = Perceptron(data=df)

percent = 80
d_split = int((percent / 100) * len(df)) # Trains on X% of data
model.train(data_split=d_split, learning_rate=0.1, epochs=20)
prediction = model.hypothesis(.85, .95)
print(f"Prediction from .85 H and .95 P: {prediction}")