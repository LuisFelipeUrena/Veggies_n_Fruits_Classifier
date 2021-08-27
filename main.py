import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
from labels import Labels #Custom made class

#Loading the Tensorflow neural netowork
model = load_model('model\Veggies_n_fruits_nn.h5')
cap = cv2.VideoCapture(0) # this opens up a webcam
lbl = Labels()

while True:
    ret, frame = cap.read()
    # cv2.imshow('frame', frame)
    resized_im = cv2.resize(frame,(100,100))
    # print(frame.shape)
    model_input = preprocess_input(resized_im)
    prediction_output = model.predict(np.array([model_input]))
    #this line below flattens the array of predictions and the index(label) with the greatest prediction
    prediction = np.argmax(np.ravel(prediction_output))
    prediction_string = lbl.getLabelName(prediction)
    #these 3 lines create a label with the name of the fruit predicted and writes
    # it into the webcam image
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(frame, prediction_string, (100,100), font,1, (0,0,0), 5, cv2.LINE_AA)
    cv2.imshow('frame', img)

    # Press the q key in your keyboard at any time to exit the program

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
