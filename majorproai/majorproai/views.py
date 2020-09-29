from django.shortcuts import render
import requests
import cv2
from tensorflow import keras

def capture():
    
    camera = cv2.VideoCapture(0)
    camera_height = 500

    while True: 
    
        a,frame = camera.read()
    
        frame = cv2.flip(frame,1)
        frame = cv2.resize(frame,(camera_height,camera_height))
    
        cv2.imshow('Capturing Frames',frame)
    
        key =cv2.waitKey(1)
    
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('classifier.jpg',frame)
            print('Image saved')

    camera.release()  
    cv2.destroyAllWindows()


def button(request):
    return render(request,'home.html')

def classifier(request):
    capture()
    categories = ["Dog","cat"]
    model = keras.models.load_model(r'C:\Progs and Concepts\AI and ML intern\Project\Model_Image_Recogniser.h5',compile = True)
    prediction = model.predict([prepare("C:\Progs and Concepts\AI and ML intern\Project\classifier.jpg")])
    final_output = str(categories[int(prediction[0][0])])
    print(categories[int(prediction[0][0])])
    return render(request,'home.html',{'output':final_output})


def prepare(filepath):
    size = 50
    img_array = cv2.imread(filepath)
    img_array = cv2.resize(img_array , (size , size))
    return img_array.reshape(-1,size,size,3)
