import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to generate noisy sine wave data
def generate_data(n_samples, noise_level):
    np.random.seed(0)
    X = np.random.rand(n_samples) * 10
    y = np.sin(X) + np.random.normal(0, noise_level, n_samples)
    return X, y

# Function to fit polynomial regression model and calculate bias-variance
def fit_polynomial(X_train, y_train, X_test, y_test, degree):
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train.reshape(-1, 1))
    X_poly_test = poly.transform(X_test.reshape(-1, 1))

    model = LinearRegression()
    model.fit(X_poly_train, y_train)

    y_pred_train = model.predict(X_poly_train)
    y_pred_test = model.predict(X_poly_test)

    bias = np.mean((y_pred_train - y_train)**2)
    variance = np.mean((y_pred_test - np.mean(y_pred_test))**2)
    
    return bias, variance, y_pred_train, y_pred_test, model

# Function to provide dynamic inferences based on bias and variance
def get_inference(bias, variance):
    if bias < variance:
        return "The model is overfitting. Try reducing model complexity or collecting more data."
    elif bias > variance:
        return "The model is underfitting. Try increasing model complexity or collecting more features."
    else:
        return "The model has a good balance between bias and variance."

# Streamlit UI
st.title('Bias-Variance Tradeoff')
st.sidebar.header('Parameters')

n_samples = st.sidebar.slider('Number of Samples', min_value=20, max_value=1000, value=50)
noise_level = st.sidebar.slider('Noise Level', min_value=0.01, max_value=1.0, value=0.1)
degree = st.sidebar.slider('Polynomial Degree', min_value=1, max_value=10, value=1)
test_size = st.sidebar.slider('Test Size', min_value=0.1, max_value=0.5, value=0.3)

X, y = generate_data(n_samples, noise_level)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

bias, variance, y_pred_train, y_pred_test, model = fit_polynomial(X_train, y_train, X_test, y_test, degree)

# Display bias and variance
st.write('### Bias and Variance')
bias_df = pd.DataFrame({'Metric': ['Bias', 'Variance'], 'Value': [bias, variance]})
st.table(bias_df)

# Dynamic inference
inference = get_inference(bias, variance)

# Plotting
fig, ax = plt.subplots()
ax.scatter(X_train, y_train, color='blue', label='Train Data')
ax.scatter(X_test, y_test, color='red', label='Test Data')

if degree <= 3:
    X_plot = np.linspace(0, 10, 1000)
    poly = PolynomialFeatures(degree=degree)
    X_poly_plot = poly.fit_transform(X_plot.reshape(-1, 1))
    y_pred_plot = model.predict(X_poly_plot)
    ax.plot(X_plot, y_pred_plot, color='green', label='Model')
else:
    ax.plot(np.sort(X_train), y_pred_train[np.argsort(X_train)], color='green', label='Model')

ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('Bias-Variance Tradeoff')
ax.legend()
st.pyplot(fig)


# Display model summary in a box
st.write('### Model Summary')
st.info(inference)
