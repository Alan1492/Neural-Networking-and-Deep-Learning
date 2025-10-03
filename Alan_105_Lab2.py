import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# Title 
st.title("Activation Function & Neural Network Explorer")

#  Visualizing Activation Functions 
st.header("1. Visualizing Activation Functions")

# Dropdown
function_name = st.selectbox(
    "Choose an activation function to display:",
    ('Step Function','Sigmoid Function','Tanh Function','ReLU Function')
)

# Define the input range that is containing 100 evenly spaced numbers from -10 to 10
x = np.linspace(-10, 10, 200)

# Activation Function Definitions 

# Step Function
def step_function(x):
    return np.where(x >= 0, 1, 0)

# Sigmoid Function (Binary)
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

# Tanh Function (Bipolar Sigmoid)
# The Tanh function is the most common type of bipolar sigmoid function.
def tanh_function(x):
    return np.tanh(x)

# ReLU Function (Rectified Linear Unit)
def relu_function(x):
    return np.maximum(0, x)

# Dictionary to map selected name to the function
functions = {
    'Step Function': step_function,
    'Sigmoid Function': sigmoid_function,
    'Tanh Function': tanh_function,
    'ReLU Function': relu_function
}

# Plotting Logic 
# Get the selected function from the dictionary
selected_function = functions[function_name]

# Calculating outputs for the selected function
y = selected_function(x)

# Visualizing the activation functions using plots
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, y, label=function_name, lw=3, color='royalblue')
ax.set_title(f'Plot of the {function_name}', fontsize=16)
ax.set_xlabel('Input Value (x)', fontsize=12)
ax.set_ylabel('Output Value', fontsize=12)
ax.legend(fontsize=12, loc='upper left')

# Add center lines for reference
ax.axhline(0, color='black', linewidth=0.7)
ax.axvline(0, color='black', linewidth=0.7)
ax.grid(True)

# Display the plot in Streamlit
st.pyplot(fig)


# XOR Problem with MLP
st.header("2. Application: Solving the XOR Problem")
st.write(
    """
    The XOR (exclusive OR) problem is a classic challenge for simple models because it's not
    linearly separable. Below, we train a Multi-Layer Perceptron (MLP) a simple neural network
    to solve it using different activation functions in its hidden layer.
    """
)

# defining dataset for the XOR problem
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# activation functions to test
# here the sigmoid function is called logistic
activation_options = ['logistic', 'tanh', 'relu']

#Loop through each activation function, train a model, and evaluate it
for activation in activation_options:
    st.subheader(f"Testing with '{activation.upper()}' Activation")

    # Create a Multi-layer Perceptron (MLP) classifier
    # hidden_layer_sizes=(4,): One hidden layer with 4 neurons.
    # solver='adam': An efficient solver for large datasets.
    # random_state=1: Ensures the results are reproducible.
    # max_iter=2500: Increased iterations to ensure the model converges.
    # i gave the iteration like 2500 because while giving the smaller values like (1000 or 2000) the model gave a warning that it ran out of training time before it could find the best solution
    model = MLPClassifier(
        hidden_layer_sizes=(4,),
        activation=activation,
        # here we can also use the lbfgs which can be efficient for smaller dataset
        solver='adam',
        random_state=1,
        max_iter=2500
    )

    # Training the model
    model.fit(X_xor, y_xor)

    # Making predictions
    predictions = model.predict(X_xor)
    
    # calculating accuracy
    accuracy = model.score(X_xor, y_xor) * 100

    # Display the results
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Inputs (X)**")
        st.dataframe(X_xor)
    with col2:
        st.write("**Outputs**")
        results_data = {
            'Actual': y_xor,
            'Predicted': predictions
        }
        st.dataframe(results_data)

    st.success(f"Accuracy: {accuracy:.2f}%")
    st.markdown("---")
