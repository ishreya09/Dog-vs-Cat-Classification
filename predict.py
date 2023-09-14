#importing dependencies
import numpy as np
import cv2

import tensorflow as tf
import tensorflow_hub as hub

# gui dependencies
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

class DogCatClassificationModel:
    def __init__(self,filename):
        mobilenet_model= "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
        pretrained_model= hub.KerasLayer(mobilenet_model,input_shape=(224,224,3), trainable=False) 
        # load model
        self.filename=filename
        self.model = tf.keras.models.load_model('model.keras',custom_objects={'KerasLayer':pretrained_model})
        self.label= self.modelPredicton()
    def getAns(self):
        return self.label
    def getModel(self):
        return self.model
    def scale_and_reshape_data(self,input_img):
        input_img_resize= cv2.resize(input_img,(224,224))
        inp_img_scaled= input_img_resize/255

        img_reshape= np.reshape(inp_img_scaled,[1,224,224,3])
        return img_reshape
    
    def read(self):
        self.input_img = cv2.imread(self.filename)
        img_reshape=self.scale_and_reshape_data(self.input_img)
        return img_reshape

    
    def predict(self):
        pred=self.model.predict(self.img)

        self.pred_label= np.argmax(pred)
        return self.pred_label
    
    def showImg(self):
        cv2.imshow("Input image",self.input_img)
        cv2.waitKey(0)
    
    def readLabel(self):
        if(self.pred_label==0):
            return "cat"
        else:
            return "dog"
    
    def modelPredicton(self):
        self.img =self.read()
        self.predict()
        ans= self.readLabel()
        return ans

class TkinterGUI:
    def __init__(self):
        self.root= Tk()
        self.root.geometry("500x400")
        self.root.title("Dog Cat Classification")
        self.root.configure(bg="black")
        self.root.resizable(False,False)
        self.createWidgets()
        self.root.mainloop()

    def createWidgets(self):
        self.makeButton("Select image (ACCEPTED FILES: .jpg, ,png)",self.makeFileDialog)
        self.filename=""
        self.predict=""

        self.makeButton("Exit",self.root.quit)
    
    def makeFileDialog(self):
        # self.root.filename = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = ("jpg"))
        self.filename = filedialog.askopenfilename(title="Select file", filetypes=[("JPEG files", "*.jpg"),("PNG files","*.png")])

        if self.filename:
            print("Selected file:", self.filename)
            self.makeLabel(self.filename)
            self.showImage()
            self.makeButton("Predict",self.make_prediction)

        else:
            print("No file selected")
        return self.filename
    
    def make_prediction(self):
        self.model=DogCatClassificationModel(self.filename)
        self.makeLabel("Prediction :"+self.model.getAns())
    
    def makeLabel(self,text):
        self.label= Label(self.root,text=text,bg="black",fg="white")
        self.label.pack()
    
    def showImage(self):
        self.img = Image.open(self.filename)
        self.img = ImageTk.PhotoImage(self.img)
        self.label= Label(self.root,image=self.img)
        self.label.pack()
    
    def makeButton(self,text,command):
        self.button= Button(self.root,text=text,command=command,bg="black",fg="white")
        self.button.pack()
    


if __name__ == "__main__":
    root=TkinterGUI()








