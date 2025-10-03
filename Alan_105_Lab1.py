import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def train_perceptron(X, y, lr=0.1, n_iter=20):
    """
    Trains a perceptron model on the given dataset.
    """
    # We start with no knowledge - weights and bias are zero. The weights determine how much importance the model gives to each input.
    # By initializing them to zeros, we are telling the model to start with the assumption that all inputs are equally unimportant.
    # It has no initial preference for input 1 or input 2.
    weights = np.zeros(X.shape[1])

    # The bias acts like a general threshold that helps the model make a decision.
    # Starting it at zero provides a neutral baseline making it neither inherently easy nor hard for the model to output a 1.
    bias = 0.0

    # The training loop runs for a specified number of iterations (epochs).
    for _ in range(n_iter):
        # Go through each example (input/output pair) in the dataset.
        for i in range(len(X)):
            # Calculating the weighted sum of inputs and adding the bias.
            prediction_raw = np.dot(X[i], weights) + bias

            # Using a step function to decide the output: 0 or 1.
            # If the raw score is positive or exactly zero, the neuron "fires," and its output is 1.
            # If the raw score is negative, the neuron does not fire, and its output is 0.
            prediction = 1 if prediction_raw >= 0 else 0

            # Figure out the error. If the prediction was right, the error is 0.
            error = y[i] - prediction

            # Update the weights and bias to correct the mistake.
            # The learning rate (lr) determines the size of the adjustment.
            # No update happens if the error is 0.
            weights += lr * error * X[i]
            bias += lr * error

    # Return what the model has learned (the final weights and bias).
    return weights, bias


# Here we are defining the dataset for all the logic gates.
# The inputs are the same for all gates.
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])

# The expected output for each gate.
gate_outputs = {
    "AND": np.array([0,0,0,1]),
    "OR":  np.array([0,1,1,1]),
    "AND-NOT": np.array([0,0,1,0]),
    "XOR": np.array([0,1,1,0]) # This is a non-linearly separable problem
}

# Storing the explanations for each gate in a dictionary.
gate_explanations = {
    "AND": """
    ### 1. How do the weights and bias values change during training for the AND gate?
    The weights and bias start at 0.0. During training, the perceptron looks at each input pair. If a mistake happens, weights and bias are adjusted to correct that error. For the AND gate, it learns that both inputs are important. The final trained values from your code can be weights like `[0.2 0.1]` and bias: `-0.2`. The positive weights show that both inputs contribute to a '1' output, while the negative bias acts as a threshold, making sure that the output is only 1 when both inputs are 1.

    ### 2. Can the perceptron successfully learn the AND logic with a linear decision boundary?
    Yes, the code shows an accuracy of 100.0%. This is because the AND gate is a linearly separable problem. You can draw a single straight line on a graph to separate the input points that result in a '0' from the one point (`[1, 1]`) that results in a '1'.
    """,
    "OR": """
    ### 3. What changes in the perceptron's weights are necessary to represent the OR gate logic?
    To represent the OR gate, the perceptron needs to learn that if at least one input is 1, the output should be 1. We can see from the code that the learned weights can be `[0.1 0.1]` and the bias: `-0.1`. Similar to the AND gate, the weights are positive. However, the bias is less negative (-0.1 for OR compared to -0.2 for AND). This lower threshold makes it easier for the neuron to activate, allowing it to output 1 even if only one input is active.

    ### 4. How does the linear decision boundary look for the OR gate classification?
    The decision boundary is a straight line that separates the `[0, 0]` input (output 0) from the other three inputs (which all output 1).
    """,
    "AND-NOT": """
    ### 5. What is the perceptron's weight configuration after training for the AND-NOT gate?
    The AND-NOT gate outputs 1 only when the first input is 1 and the second input is 0. The weights obtained while training can be `[0.1 -0.2]` and bias: `-0.1`. Notice that the weight for the first input is positive (encouraging a 1 output) while the weight for the second input is negative (discouraging a 1 output).

    ### 6. How does the perceptron handle cases where both inputs are 1 or 0?
    It handles them correctly by using the negative weight for the second input.
    * **Input [1, 1]:** The negative weight (-0.2) for the second input can cancel out the positive weight from the first, resulting in a negative sum. The prediction is correctly 0.
    * **Input [0, 0]:** With no positive input, the negative bias ensures the result is negative, predicting 0.
    """,
    "XOR": """
    ### 7. Why does the Single Layer Perceptron struggle to classify the XOR gate?
    The Single Layer Perceptron fails because the XOR gate is not linearly separable. This means we cannot draw a single straight line to separate the points that should output 0 (`[0,0]` and `[1,1]`) from the points that should output 1 (`[0,1]` and `[1,0]`). A single perceptron can only create one straight line, so it becomes impossible to solve this problem. This is why the accuracy is only 50%.

    ### 8. What modifications can be made to the neural network model to handle the XOR gate correctly?
    To solve the XOR problem, the model needs to be able to create a more complex, non-linear decision boundary. This can be achieved in two main ways:
    * **Adding More Layers:** Instead of a single-layer perceptron, you can use a Multi-Layer Perceptron (MLP). By adding a "hidden layer" of neurons between the input and output, the network can combine simple lines to form a more complex shape that correctly separates the XOR data.
    * **Using Non-Linear Activation Functions:** The simple step function used in the perceptron is linear. Using an advanced, non-linear activation function like Sigmoid or ReLU in a multi-layer network allows it to learn non-linear relationships like the one in the XOR gate.
    """
}

# Streamlit Interface 
st.title("Perceptron Logic Gate Simulator")

# Dropdown menu to select the logic gate.
gate_name = st.selectbox("Choose a Logic Gate:", list(gate_outputs.keys()))
outputs = gate_outputs[gate_name]

# Sliders to adjust the hyperparameters for training.
# The learning rate (lr) determines how big of an adjustment the model makes after each error.
lr = st.slider("Learning Rate (lr)", min_value=0.01, max_value=1.0, step=0.01, value=0.1)
# The number of iterations (epochs) tells how many times the learning algorithm will work through the entire training dataset.
epochs = st.slider("Number of Epochs", min_value=1, max_value=100, step=1, value=20)

# Train the perceptron with the selected gate and hyperparameters.
weights, bias = train_perceptron(inputs, outputs, lr=lr, n_iter=epochs)

st.write(f"**Learned Weights:** {np.round(weights,2)}")
st.write(f"**Learned Bias:** {np.round(bias,2)}")

#  Testing and Results 
st.subheader("Testing Results")
correct_count = 0
# Observing how well the model has learned by testing it on the same inputs.
for i in range(len(inputs)):
    # Make a prediction using the learned weights and bias.
    final_prediction_raw = np.dot(inputs[i], weights) + bias
    final_prediction = 1 if final_prediction_raw >= 0 else 0

    if final_prediction == outputs[i]:
        correct_count += 1
    st.write(f"Input {inputs[i]} ===>Predicted: {final_prediction} (Actual: {outputs[i]})")

# Calculating the accuracy of the model: the percentage of time it predicted the correct answer.
accuracy = (correct_count / len(inputs)) * 100
st.success(f"Accuracy: {accuracy:.2f}%")

# Special Visualization for XOR 
# This plot shows why the single-layer perceptron fails on the XOR problem.
if gate_name == "XOR":
    y_xor = outputs
    fig, ax = plt.subplots(figsize=(5,4))
    # Plotting points with output 0 in red and output 1 in blue.
    ax.scatter(inputs[y_xor==0,0], inputs[y_xor==0,1], c='red', marker='o', s=100, label='Output 0')
    ax.scatter(inputs[y_xor==1,0], inputs[y_xor==1,1], c='blue', marker='x', s=100, label='Output 1')
    ax.set_title("XOR Problem Space")
    ax.set_xlabel("Input A")
    ax.set_ylabel("Input B")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.info("A plot to show the XOR problem. The reason for the failure is that the perceptron can't find a single straight line to separate the 0s from the 1s. In simple words, this is a classic example of a non-linearly separable problem.")

# Explanations for each gate
st.subheader(f"Explanation for {gate_name} Gate")
# Display the explanation text for the currently selected gate.
st.markdown(gate_explanations[gate_name])
