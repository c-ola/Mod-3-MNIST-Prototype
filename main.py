from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import torch
import combined_model_final
import mnist

test_images = mnist.test_images()
test_labels = mnist.test_labels()

combined = combined_model_final.Combined(test_images, test_labels)

#setting up tkinter
root = Tk()
root.title("Machine Learning Models: Accuracy Testing")
root.geometry("1200x700")

#returns what image to use when testing (text for output)
def getImage():
    text = ""
    img = None

    if use_drawing.get():
        img = imgdata 
        text = ""
    elif index.get() != "" and (int(index.get()) > 0 and int(index.get()) <= len(test_images)):
        img = test_images[int(index.get())-1]
        text = f"Real label: {test_labels[int(index.get())-1]}\n"
        resetCanvas()
    else:
        text = "Invalid Input"

    return img, text

#draws the output on the interface as well as the number on the canvas
def draw_output(img, text):
  
    output = Label(root, text=text, padx=40, pady=10)
    output.grid(row=3, column=2)  

    if not img is None:
        for j in range(28):
            for i in range(28):
                color = rgb_hack((int(img[i, j]), int(img[i, j]), int(img[i, j])))
                canvas.create_rectangle(j*10, i*10, j*10+10, i*10+10, outline=color, fill=color) 

#outputs test prediction for KNN
def run_knn():
    (img, text) = getImage()

    if not img is None:
        x = int(combined.knn_pred(torch.from_numpy(img)))
        text = text + f"KNN prediction: {x}"

    draw_output(img, text)

#outputs test prediction for naive bayes
def run_nbc():
    (img, text) = getImage()

    if not img is None:
        x = combined.naive_bayes_pred(torch.from_numpy(img))
        text = text + f"Naive Bayes prediction: {x}"

    draw_output(img, text)

#outputs test prediction for CNN
def run_cnn():
    (img, text) = getImage()
    
    if not img is None:
        x = combined.cnn_pred(torch.from_numpy(img)).argmax()
        text = text + f"CNN prediction: {x}"

    draw_output(img, text)

#outputs combined test prediction
def run_combined():
    (img, text) = getImage()

    if not img is None:
        x = combined.combined_out(torch.from_numpy(img))
        text = text + f"Combined prediction: {x}"

    draw_output(img, text)

#converts (red, green, blue) to hex
def rgb_hack(rgb):
    return "#%02x%02x%02x" % rgb

#scales a value to a range
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

#drawing handwritten digits on the canvas
def drawOnCanvas(e):
    root.update()
    x, y = e.x, e.y

    r = int(brush_width/2)

    sx = x/10
    sy = y/10

    for i in range(int(sx), r + int(sx)):
        for j in range(int(sy), r + int(sy)):
            d2 = (i-sx)**2+(j-sy)**2

            #change last 2 values to change color range
            base_color = 255 - int(scale(d2, 0, r**2, 0, 255))                

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

#canvas stuff
brush_width = 4
canvas_width = 280
canvas_height = 280
use_drawing = IntVar()
imgdata = np.zeros((28, 28)) #keep track of color on canvas
canvas = Canvas(root, width=canvas_width, height=canvas_height, bg="black")
canvas.grid()
canvas.bind("<ButtonPress-1>", drawOnCanvas)
canvas.bind("<B1-Motion>", drawOnCanvas)

#buttons regarding the canvas
button_inc = Button(root, text="Increase Brush Size", padx=20, pady=20, command=increaseBrushWidth)
button_inc.grid(row=1, column=0)
button_dec = Button(root, text="Decrease Brush Size", padx=20, pady=20, command=decreaseBrushWidth)
button_dec.grid(row=2, column=0)
button_reset_canvas = Button(root, text="Reset Canvas", padx=20, pady=20, command=resetCanvas)
button_reset_canvas.grid(row=3, column=0)
check_drawing = Checkbutton(root, text='Use Drawing for Tests', variable=use_drawing)
check_drawing.grid(row=4, column=0)

#index for test image
promptIndex = Label(root, text="Enter a number from 1 to 10000\n This value is the test image's index in the MNIST dataset")
promptIndex.grid(row=1, column=2)
index = Entry(root, width=10)
index.grid(row=2, column=2)

#define all of the buttons size, text, and function which they call
button_knn = Button(root, text="Test K-Nearest Neighbour", padx=40, pady=20, command=run_knn)
button_knn.grid(row=1, column=4)
button_cnn = Button(root, text="Test Convolutional Neural Network", padx=40, pady=20, command=run_cnn)
button_cnn.grid(row=2, column=4)
button_nbc = Button(root, text="Test Naive Bayes Classifier", padx=40, pady=20, command=run_nbc)
button_nbc.grid(row=3, column=4)
button_combined = Button(root, text="Test combination of all models", padx=40, pady=20, command=run_combined)
button_combined.grid(row=4, column=4)
button_quit = Button(root, text="Exit", padx=40, pady=40, command=root.quit)
button_quit.grid(row=5, column=4)

#Blank boxes for cell formatting
blank1 = Label(root, text=" ", padx=80)
blank2 = Label(root, text=" ", padx=80)
blank3 = Label(root, text=" ", padx=80)

blank1.grid(row=0, column=1)
blank2.grid(row=0, column=2)
blank3.grid(row=0, column=3)


root.mainloop()

