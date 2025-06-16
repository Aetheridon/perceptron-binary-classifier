# Perceptron Binary Classifier

A simple binary classifier that predicts whether it will rain, using a **Perceptron algorithm** built from scratch in Python.

This project uses a mock weather dataset with two features — **humidity** and **pressure** — to train the model to classify days as either **"Rain" (`1`)** or **"No Rain" (`0`)**.

The model works by:
- Training **weights** and a **bias** using the Perceptron learning rule
- Applying these learned values to a linear equation that calculates a **score** from the inputs
- Passing the score through a **step function** to generate a final binary classification (`1` or `0`)
