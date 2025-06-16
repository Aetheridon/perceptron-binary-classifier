from perceptron import Perceptron
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("mock_weather_data.csv")
df.dropna(inplace=True)
model = Perceptron(data=df)

# Training
percent = 80
d_split = int((percent / 100) * len(df)) # Trains on X% of data
model.train(data_split=d_split, learning_rate=0.1, epochs=20)

# Predicting on test data
test_data = df[d_split:]
for i, row in test_data.iterrows():
    humidity = row["humidity"]
    pressure = row["pressure"]
    r_label = row["rain"]

    prediction = model.hypothesis(humidity, pressure)
    print(f"Humidity: {humidity}, Pressure: {pressure} -> Predicted: {prediction}, Actual: {r_label}")

# Plotting results
for label, color in zip([0, 1], ["black", "blue"]):
    subset = test_data[test_data.apply(
        lambda row: model.hypothesis(row["humidity"], row["pressure"]) == label,
        axis=1
    )]

    plt.scatter(subset["humidity"], subset["pressure"], color=color, label=f"Predicted Rain={label}", alpha=0.6)

plt.xlabel("Humidity")
plt.ylabel("Pressure")
plt.title("Perceptron Predictions on Test Data")
plt.legend()
plt.grid(True)
plt.savefig("perceptron_predictions.png")