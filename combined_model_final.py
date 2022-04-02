import torch
import torch.nn as nn
import torch.nn.functional as F
import mnist
import numpy as np
import matplotlib.pyplot as plt

torch.random.manual_seed(0)

train_images = mnist.train_images()
train_images = torch.tensor(train_images, dtype=torch.float64).reshape(60000, 1, 28, 28)
train_labels = torch.tensor(mnist.train_labels(), dtype=torch.int64).reshape(60000, )

test_images = mnist.test_images()
test_images = torch.tensor(test_images, dtype=torch.float64).reshape(10000, 1, 28, 28)
test_labels = torch.tensor(mnist.test_labels(), dtype=torch.int64).reshape(10000, )

class Cnn(nn.Module):


  def __init__(self):

    super(Cnn, self).__init__()

    self.conv1 = nn.Conv2d(1, 10, 3)
    self.pool = nn.MaxPool2d((2, 2))
    self.conv2 = nn.Conv2d(10, 50, 3)
    self.lin1 = nn.Linear(6050, 1000)
    self.lin2 = nn.Linear(1000, 10)
  
  
  def forward(self, input):

    input = input.view(-1, 1, 28, 28).float()

    input = F.relu(self.conv1(input))
    input = F.relu(self.pool(input))
    input = F.relu(self.conv2(input))

    input = input.view(-1, 6050)
    input = F.relu(self.lin1(input))
    input = self.lin2(input)

    return input


  def test(self, test_set, test_labels):

    acc = 0

    for i, input in enumerate(test_set):
      pred = self.forward(input)
      pred = int(pred.argmax())
      
      correct = test_labels[i]
      if pred == correct:
        acc += 1
    
    print(f'accuracy: {acc}/{len(test_labels)} = {round(acc / len(test_labels), 3) * 100}')

cnn = Cnn()

cnn.load_state_dict(torch.load('Pytorch MNIST CNN'))



"""#Naive Bayes"""

from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

#seperates and loads data from mnist into training and testing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#x = number of pixels for each image
#y = actual number of the digit
trainNum = 20000
testNum = 10000
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

#setting up multionmial naive bayes, fitting and seeing the overall accuracy of the model
mnb = MultinomialNB()
mnb.fit(x_train,y_train)
mnb.score(x_train,y_train)


"""#KNN"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#download MNIST dataset
keras.datasets.mnist.load_data(path="mnist.npz")
(trImages, trLabels), (tImages, tLabels) = keras.datasets.mnist.load_data()
numTrainImages = np.shape(trLabels)[0]
numTestImages = np.shape(tLabels)[0]

paramk = 11 #choose a parameter for k
numErrs = 0

num = 0

arrayKNNLabels = np.array([])
arrayDistance = np.array([]) #used to store distance of a test image from all training images
for j in range(numTrainImages):  
  distance = np.sum((np.sqrt((trImages[j]-tImages[num])/255.0)**2)) #distance between two images; divide by 255 to normalize data (255 is the max pixel value)  
  arrayDistance = np.append(arrayDistance, distance)
    
sort = np.argsort(arrayDistance) #sorts distances to easily find the shortest ones
  
kLabels = trLabels[sort[0:paramk]] #choose first k labels  
(values, counts) = np.unique(kLabels, return_counts=True) #gets the labels and counts them
arrayKNNLabels = np.append(arrayKNNLabels, values[np.argmax(counts)])#gets the most common label


"""#Assembling the models into one class where they can all look at one image at once


"""

x = 0  # this is the index of the image we would like to test the networks on

input_image = test_images[x]


class Combined():


  def cnn_pred(self, image):

    prediction = cnn.forward(image)

    return prediction # outputs a 1x10 tensor of 


  def naive_bayes_pred(self, image):
    image = image.reshape(28*28, ).numpy().astype(int)

    prediction = mnb.predict([image])

    return int(prediction)


  def knn_pred(self, image):
    image = image.reshape(28, 28).numpy()

    arrayKNNLabels = np.array([])
    arrayDistance = np.array([]) #used to store distance of a test image from all training images
    for j in range(numTrainImages):  
      distance = np.sum((np.sqrt((abs(trImages[j]-image))/255.0)**2)) #distance between two images; divide by 255 to normalize data (255 is the max pixel value)  
      arrayDistance = np.append(arrayDistance, distance)
    
    sort = np.argsort(arrayDistance) #sorts distances to easily find the shortest ones
  
    kLabels = trLabels[sort[0:paramk]] #choose first k labels  
    (values, counts) = np.unique(kLabels, return_counts=True) #gets the labels and counts them
    arrayKNNLabels = np.append(arrayKNNLabels, values[np.argmax(counts)])#gets the most common label

    prediction = arrayKNNLabels[-1]

    return int(prediction)


  def combined_out(self, image):

    cnn_out = int(self.cnn_pred(image).argmax())
    bayes_out = int(self.naive_bayes_pred(image))
    knn_out = int(self.knn_pred(image))

    points_cnn = 1 # bias towards most accurate network
    points_bayes = 0
    points_knn = 0

    #adding points if the network agrees with another network
    if cnn_out == bayes_out:
      points_cnn += 1
      points_bayes += 1
    if cnn_out == knn_out:
      points_cnn += 1
      points_knn += 1
    if bayes_out == knn_out:
      points_bayes += 1
      points_knn += 1
    
    output = cnn_out
    output = bayes_out if points_bayes > points_cnn and points_bayes > points_knn else output
    output = knn_out if points_knn > points_cnn and points_bayes > points_knn else output
    return output


model = Combined()


def test(num_images): 
  
  acc = 0
  
  for i in range(num_images):
    image = test_images[i]

    pred = model.combined_out(image)
    if int(pred) == int(test_labels[i]):
      acc += 1
    if (i + 1) % 100 == 0:
      print(f"{i} images tested, current accuracy {float(acc) / float(i) * 100}% ({acc}/{i})")

  print(f"Final accuracy: {float(acc) / float(i) * 100}%")
  print(f"{acc} / {num_images} images guessed correctly")
