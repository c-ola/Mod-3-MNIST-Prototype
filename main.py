from tkinter import *
import numpy as np
import matplotlib.pyplot as plot
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split
import torch

import combined_model_final
import mnist

test_images = mnist.test_images()
test_labels = mnist.test_labels()

combined = combined_model_final.Combined(test_images, test_labels)


# import numpy as np
import matplotlib.pyplot as plt

root = Tk()
root.title("Machine Learning Models: Accuracy Testing")
root.geometry("1200x700")

def run_knn():
    print(combined.knn_pred(7))
    # plt.imshow(test_images[7].reshape(28, 28))
    # plt.show()
    def knn_predict():
        x = combined.knn_pred(int(index.get()))
        print(x)

        output = Label(root, text=f"Real label: {test_labels[int(index.get())]}\nKNN predicted label: {x}")
        output.grid(row=3, column=1)

        plt.imshow(test_images[int(index.get())].reshape(28, 28))
        plt.show()

    index = Entry(root, width=10)
    index.grid(row=2, column=1)

    button_index = Button(root, text="Enter an index for the MNIST dataset", padx=40, pady=20, command=knn_predict)
    button_index.grid(row=1, column=1)

def run_nbc():
    def nbc_predict():
        x = combined.naive_bayes_pred(int(index.get()))
        print(x)

        output = Label(root, text=f"Real label: {test_labels[int(index.get())]}\nPredicted label: {x}")
        output.grid(row=3, column=1)

        plt.imshow(test_images[int(index.get())].reshape(28, 28))
        plt.show()

    index = Entry(root, width=10)
    index.grid(row=2, column=1)

    button_index = Button(root, text="Enter an index for the MNIST dataset", padx=40, pady=20, command=nbc_predict)
    button_index.grid(row=1, column=1)







def run_cnn():
    def cnn_predict():
        #x = combined.cnn_pred(int(index.get().argmax()))

        x = int(combined.cnn_pred(int(index.get())).argmax())

        print(x)

        output = Label(root, text=f"Real label: {test_labels[int(index.get())]}\nPredicted label: {x}")
        output.grid(row=3, column=1)

        plt.imshow(test_images[int(index.get())].reshape(28, 28))
        plt.show()

    index = Entry(root, width=10)
    index.grid(row=2, column=1)

    button_index = Button(root, text="Enter an index for the MNIST dataset", padx=40, pady=20, command=cnn_predict)
    button_index.grid(row=1, column=1)
    return

def run_combined():
    print(combined.combined_out(7))
    return

def rgb_hack(rgb):
    return "#%02x%02x%02x" % rgb

def scale(r, rmin, rmax, tmin, tmax):
    return ((r - rmin)/(rmax - rmin)) * (tmax -tmin) + tmin

def decreaseBrushWidth():
    global brush_width
    minBrush = 1
    if brush_width > minBrush + 1:
        brush_width-=1

def increaseBrushWidth():
    global brush_width
    maxBrush = 20
    if brush_width < maxBrush:
        brush_width+=1

def resetCanvas():
    global imgdata
    root.update()
    imgdata = np.zeros((28, 28))    
    canvas.delete('all')

def drawOnCanvas(e):
    root.update()
    x, y = e.x, e.y

    r = int(brush_width/2)

    sx = x/10
    sy = y/10

    for i in range(int(sx), r + int(sx)):
        for j in range(int(sy), r + int(sy)):
            d2 = (i-sx)**2+(j-sy)**2
            base_color = int(255 - scale(d2, 0, r**2, 0, 255))                

            #faster way to draw a circle using its symmetries
            #r^2 instead of (r^2)^2
            xpoints =[int(i), int(2*sx-i), int(2*sx-i), int(i)]

            ypoints =[int(j), int(2*sy-j), int(j), int(2*sy-j)]
        
            color_at_symmetry =[
            imgdata[ypoints[0], xpoints[0]],
            imgdata[ypoints[1], xpoints[1]],
            imgdata[ypoints[2], xpoints[2]],
            imgdata[ypoints[3], xpoints[3]]
            ]       
            
            if d2 < r**2:
                for k in range(4):                    
                    color = ((int(base_color), int(base_color), int(base_color)))      
                    if base_color > color_at_symmetry[k]:
                        canvas.create_rectangle(xpoints[k]*10, ypoints[k]*10, xpoints[k]*10+10, ypoints[k]*10+10, outline=rgb_hack(color), fill=rgb_hack(color))  
                        imgdata[ypoints[k], xpoints[k]] = base_color


def test_drawing():
    print(np.array(imgdata).reshape(1, 28, 28))
    model = combined_model_final.Combined(np.array(imgdata).reshape(1, 28, 28), [0])
    print(f"Prediction for drawing: {model.combined_out(0)}")

#canvas stuff
brush_width = 5
canvas_width = 280
canvas_height = 280
imgdata = np.zeros((28, 28)) #keep track of color on canvas
canvas = Canvas(root, width=canvas_width, height=canvas_height, bg="black")
canvas.grid(row=0, column=0)
canvas.bind("<ButtonPress-1>", drawOnCanvas)
canvas.bind("<B1-Motion>", drawOnCanvas)

#buttons regarding the canvas
button_inc = Button(root, text="Increase Brush Size", padx=20, pady=20, command=increaseBrushWidth)
button_inc.grid(row=1, column=0)
button_dec = Button(root, text="Decrease Brush Size", padx=20, pady=20, command=decreaseBrushWidth)
button_dec.grid(row=2, column=0)
button_reset_canvas = Button(root, text="Reset Canvas", padx=20, pady=20, command=resetCanvas)
button_reset_canvas.grid(row=3, column=0)

#define all of the buttons size, text, and function which they call
button_knn = Button(root, text="Test K-Nearest Neighbour", padx=40, pady=20, command=run_knn)
button_cnn = Button(root, text="Test Convolutional Neural Network", padx=40, pady=20, command=run_cnn)
button_nbc = Button(root, text="Test Naive Bayes Classifier", padx=40, pady=20, command=run_nbc)
button_combined = Button(root, text="Test combination of all models", padx=40, pady=20, command=run_combined)

button_testDrawing = Button(root, text="Test canvas", padx=40, pady=20, command=test_drawing)

button_quit = Button(root, text="Exit", padx=40, pady=40, command=root.quit)

#put the buttons on the screen

button_knn.grid(row=1, column=4)
button_cnn.grid(row=2, column=4)
button_nbc.grid(row=3, column=4)
button_combined.grid(row=4, column=4)

button_testDrawing.grid(row=4, column=0)

button_quit.grid(row=5, column=4)

#Blank boxes for cell formatting

blank1 = Label(root, text=" ", padx=80)
blank2 = Label(root, text=" ", padx=80)
blank3 = Label(root, text=" ", padx=80)


blank1.grid(row=0, column=1)
blank2.grid(row=0, column=2)
blank3.grid(row=0, column=3)


root.mainloop()

