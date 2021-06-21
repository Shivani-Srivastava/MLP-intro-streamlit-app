# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 09:38:27 2021

@author: 20052
"""
# https://github.com/sudhir-voleti/MLBM/blob/master/Lec%2009a%20Intro%20to%20ANNs%20in%20sklearn.ipynb

## setup chunk
#!pip install mglearn
#!pip install graphviz
import streamlit as st
import mglearn
import graphviz

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression

#%matplotlib inline
import matplotlib.pyplot as plt

import numpy as np
import time


# define funcs
def show_activations():
    line = np.linspace(-3, 3, 100)
    sigmoid1 = 1/(1+np.exp(-1*line))  # defining sigmoid func

    # plot
    plt.plot(line, np.tanh(line), 'b-', label="tanh")  # tanH func
    plt.plot(line, np.maximum(line, 0), 'r--', label="relu")  # reLU func
    plt.plot(line, sigmoid1, 'g-', label="sigmoid")  # sigmoid

    plt.legend(loc="best")
    plt.xlabel("x")
    plt.ylabel("relu(x), tanh(x), sigmoid(x)")
    plt.grid(True)
    plt.show()
    
    

# simulate dataset. parm=n_samples1
def simulate_data(n_samples1=100):
    X, y = make_moons(n_samples=n_samples1, noise=0.25, random_state=3)

    # view dataset plot
    plt.plot(X[(y==0),0], X[(y==0),1], 'bo', label="class 0")
    plt.plot(X[(y==1),0], X[(y==1),1], 'r^', label="class 1")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.legend()
    plt.show()    
    
# build n run an MLP
def build_run_MLP(n_samples1, hidden_layer_sizes1, # default is [100]. Try [10, 10] etc
                  activation1): # default is 'relu', else 'tanh', 'logistic', 'identity'

    X, y = make_moons(n_samples=n_samples1, noise=0.25, random_state=3)
    
    # stratified train-test-split with set random seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    
    #hidden_layer_sizes2 = [int(x) for x in hidden_layer_sizes1]

    # invoke, instantiate and fit to data the MLP classifier with parm from above
    mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes = hidden_layer_sizes1, activation=activation1).fit(X_train, y_train)
    return mlp, X_train, y_train, X_test, y_test
    
    
def main():
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.title("Welcome to an MLP Streamlit app")
	st.write("Aim is to demo an ANN's basic compts. Select one option to proceed:")
    
	menu = ["Show Activation Funcs", "Show data", "Run MLP"]
	choice = st.sidebar.selectbox("Menu", menu)

	if choice == "Show Activation Funcs":
		st.markdown("### Activation Functions")
		st.write("Nonlinear functions that map weighted observations to outcomes. Come in 3 main flavors, as below.")
		fig = show_activations()
		st.pyplot(fig)

	if choice == "Show data":
		st.markdown("### Building Simulated Data")
		st.write("Choose how many data points to simulate in 2-D. Default is 100.")
		n_samples1 = st.slider("Select the level", 100, 500)
		st.text('Selected: {}'.format(n_samples1))
		st.pyplot(simulate_data(n_samples1))

	if choice == "Run MLP":
		st.markdown("### Building a simple Multi-Layer_perceptron (MLP)")
        
		st.write("Choose how many data points to simulate in 2-D. Default is 100.")
		n_samples1 = st.slider("Select the level", 100, 500)        
		st.text('Selected: {}'.format(n_samples1))
        
		st.text("Below, enter hiddenlayer sizes in square brackets. E.g., [100,10]")
		hidden_layer_sizes1 = st.selectbox("Options: ", [100, 10, [10,10], [100,100]])
        
		st.text("Choose an activation layer")
		activation1 = st.selectbox("Options: ", ['relu', 'logistic', 'tanh'])
		
		mlp, X_train, y_train, X_test, y_test = build_run_MLP(n_samples1, hidden_layer_sizes1, activation1)
    
		st.write("Specs of the MLP: ", mlp, "\n")    
    
		# use mglearn plots to visualize results
		mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
		mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
		plt.xlabel("Feature 0")
		plt.ylabel("Feature 1")
		st.pyplot()

		st.markdown('### Evaluating MLP perf')
		st.write("MLP Training set score: ", round(mlp.score(X_train, y_train), 3), "\n")
		st.write("MLP Test set score: ", round(mlp.score(X_test, y_test), 3), "\n")
    
		st.markdown('### Compare perf with the std Logistic Classifier')
		logreg = LogisticRegression(random_state=0).fit(X_train, y_train)
		st.write("LogReg Training set score: ", round(logreg.score(X_train, y_train), 3), "\n")
		st.write("LogReg Test set score: ", round(logreg.score(X_test, y_test), 3), "\n")


if __name__ == '__main__':
    main()    