# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 12:51:31 2021

@author: 20052
"""
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
 
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

## --- funcs
# Visualization
def plot_gallery(images, titles, h, w, rows=3, cols=4):
    plt.figure(figsize=(15,10))
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())
 
def titles(y_pred, y_test, target_names):
    for i in range(y_pred.shape[0]):
        pred_name = target_names[y_pred[i]].split(' ')[-1]
        true_name = target_names[y_test[i]].split(' ')[-1]
        yield 'predicted: {0}\ntrue: {1}'.format(pred_name, true_name)
 
 
# Load data
min_faces_per_person_ui=100
lfw_dataset = fetch_lfw_people(min_faces_per_person=min_faces_per_person_ui) # 5.8s
 
_, h, w = lfw_dataset.images.shape
X = lfw_dataset.data
y = lfw_dataset.target
target_names = lfw_dataset.target_names
print(target_names)
 
# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Compute a PCA 
n_components_ui = 100  # from UI
pca = PCA(n_components=n_components_ui, whiten=True).fit(X_train)
 
# apply PCA transformation
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# train a neural network
print("Fitting the classifier to the training set")
clf = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True).fit(X_train_pca, y_train) # 2s

y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=target_names))

 
prediction_titles = list(titles(y_pred, y_test, target_names))
plot_gallery(X_test, prediction_titles, h, w)

## main
def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown("## ML on Images - the LFW dataset")
    st.text("make the 2 parm selections below")
    
    min_faces_per_person_ui = st.slider("Select Min num of Faces per person", 50, 200)
                                        
    lfw_dataset = fetch_lfw_people(min_faces_per_person=min_faces_per_person_ui) # 5.8s
    _, h, w = lfw_dataset.images.shape
    X = lfw_dataset.data
    y = lfw_dataset.target
    target_names = lfw_dataset.target_names
    st.write(list(target_names))    
    
	# split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Compute a PCA 
    n_components_ui = st.slider("Select Num of PCA compts", 50, 500)    
    pca = PCA(n_components=n_components_ui, whiten=True).fit(X_train)
 
    # apply PCA transformation
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)    
    
    # train a neural network
    #print("Fitting the classifier to the training set")
    clf = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True).fit(X_train_pca, y_train) # 2s

    y_pred = clf.predict(X_test_pca)
    st.write(multilabel_confusion_matrix(y_test, y_pred))
    a0 = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    a1 = pd.DataFrame.from_dict(a0)
    st.write(a1)
 
    prediction_titles = list(titles(y_pred, y_test, target_names))
    st.pyplot(plot_gallery(X_test, prediction_titles, h, w))
    

if __name__ == "__main__":
    main()
