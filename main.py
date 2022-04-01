#changes for github testing
##
#
#
#
a = 0
from tkinter import *
import numpy as np
import matplotlib.pyplot as plot
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from tensorflow.keras.datasets import mnist
from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split
import torch

import combined_model_final

combined = combined_model_final.Combined()


# import mnist
# import numpy as np
import matplotlib.pyplot as plt

root = Tk()
root.title("Machine Learning Models: Accuracy Testing")

canvas = Canvas(root, width = 400, height = 400)
canvas.pack()

def run_knn():

  '''# #loading data
  (X_train, y_train), (X_test, y_test) = mnist.load_data()
  # #reshaping data
  X_train = X_train[:60000]
  y_train = y_train[:60000]
  X_test = X_test[:1000]
  y_test = y_test[:1000]
  #
  X_train = X_train.reshape((60000, 28 * 28))
  X_test = X_test.reshape((1000, 28 * 28))
  #
  num = int(input("Enter number:"))
  #
  #creating and training knn classifier
  knn = neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform')
  knn.fit(X_train, y_train)
  #
  #getting a prediction
  pred = knn.predict(X_test)
  plt.imshow(X_test[num].reshape(28, 28))
  print("Real Label:", y_test[num])
  plt.show()
  print(f"Predicted Label: {int(combined.knn_pred(X_test[num]))}")
'''
  print("ran function")

def run_nbc():
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    trainNum = 20000
    testNum = 10000
    xTrain = xTrain.reshape(60000, 28 * 28)
    xTest = xTest.reshape(10000, 28 * 28)
    NBC_training = Label(root, text='Begin Training')
    #NBC_training.grid(row=1, column=2)
    naive_bayes_multinomial = MultinomialNB()
    fit_multinomial = naive_bayes_multinomial.fit(xTrain, yTrain)
    predictions = fit_multinomial.predict(xTest)
    con_matrix_multinomial = confusion_matrix(yTest, predictions)
    print("Finished Training\n")

    print("Begin Testing")
    print(fit_multinomial.score(xTest, yTest))
    print(con_matrix_multinomial)
    print("Finish Testing")


    return
#nbc code


def run_cnn():
    return


def run_combined():
    return

def drawOnCanvas(e):
    x, y = e.x, e.y
    canvas.create_line(x, y, x+1, y+1)



#define all of the buttons size, text, and function which they call
button_knn = Button(root, text="Test K-Nearest Neighbour", padx=40, pady=20, command=run_knn)
button_cnn = Button(root, text="Test Convolutional Neural Network", padx=40, pady=20, command=run_cnn)
button_nbc = Button(root, text="Test Naive Bayes Classifier", padx=40, pady=20, command=run_nbc)
button_combined = Button(root, text="Test combination of all models", padx=40, pady=20, command=run_combined)

button_quit = Button(root, text="Exit", padx=40, pady=20, command=root.quit)
root.bind('<Motion>', drawOnCanvas)

#put the buttons on the screen
button_knn.grid(row=0, column=0)
button_cnn.grid(row=0, column=1)
button_nbc.grid(row=0, column=2)
button_combined.grid(row=0, column=3)

button_quit.grid(row=5, column=3)

#Text boxes

'''#drawing stuff
canvas = tk.Canvas(root, width=400, height=400)
canvas.pack()
'''

root.mainloop()

