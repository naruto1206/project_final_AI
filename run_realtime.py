
import string
from cv2 import CAP_MSMF
from keras.models import load_model
from time import sleep
from keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import time

prediction = ''
score = 0
bigModel = None

gesture_names = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'a',11:'b',12:'c',13:'d',14:'e',15:'f',16:'g',17:'h',18:'i',19:'j',20:'k',21:'l',22:'m',23:'n',24:'o',25:'p',
26:'q',27:'r',28:'s',29:'t',30:'u',31:'v',32:'w',33:'x',34:'y',35:'z'}

model = load_model('handsign_recognition_30epochs.h5')

def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32')
    image /=255
    pred_array = model.predict(image)
    print(f'pred_array: {pred_array}')
    result = gesture_names[np.argmax(pred_array)]
    print(f'Result: {result}')
    print(max(pred_array[0]))
    score = float("%0.2f" % (max(pred_array[0])*100))
    print(result)
    return result, score


def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask = fgmask)
    return res

cap_region_x_begin = 0.5
cap_region_y_end = 0.8

threshold = 60
blurValue = 41
bgSubThreshold = 50
learningRate = 0

predThreshold= 95

isBgCaptures = 0

camera = cv2.VideoCapture(0)
camera.set(10,200)
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.01)

while camera.isOpened():
    ret, frame = camera.read()
    frame = cv2.bilatealFilter(frame, 5, 50, 100)

    frame = cv2.flip(frame, 1)


    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]),0 ),
                (frame.shape[1], int(cap_region_y_end*frame.shape[0])), (250, 0 ,0),2 )
    
    if isBgCaptures == 1:
        img = remove_background(frame)

        img = img[0:int(cap_region_y_end*frame.shape[0]),
        int(cap_region_x_begin*frame.shape[1]):frame.shape[1]]


        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)

        cv2.imshow('original1', cv2.resize(blur, dsize = None, fx=0.5, fy=0.5))

        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cv2.imshow('thresh', cv2.resize(thresh, dsize=None, fx=0.5, fy = 0.5))

        if (np.count_nonzero(thresh)/(thresh.shape[0]*thresh.shape[0])>0.2):
            if (thresh is not None):
                target = np.stack((thresh,) * 3, axis=-1)
                target = cv2.resize(target, (224, 224))
                target = target.reshape(1, 224, 224, 3)
                prediction, score = predict_rgb_image_vgg(target)


                print(score,prediction)
                if(score>=predThreshold):
                    cv2.putText(frame, "Sign" + prediction, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, 
                    (0, 0, 255), 10, lineType=cv2.LINE_AA)
    thresh = None

    k = cv2.waitKey(10)
    if k == ord('q'): #bam q de thaot
        break
    elif k == ord('b'):
        bgModel = cv2.createBackgroundSubtractorM0G2(0, bgSubThreshold)

        isBgCaptured = 1
        cv2.putText(frame, "Background captured", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3,
        (0, 0, 255), 10, lineType=cv2.LINE_AA)
        time.sleep(2)
        print('Background captures')
    
    elif k == ord('r'):

        bgModel = None
        isBgCaptures = 0
        cv2.putText(frame, "Background reset", (20, 150), cv2.FONT_HERHSHEY_SIMPLEX, 3,
        (0, 0, 255), 10, lineType=cv2.LINE_AA)
        print('Background reset')
        time.sleep(1)

    cv2.imshow('original', cv2.resize(frame, dsize= None, fx=0.5, fy=0.5))

cv2.destroyAllWindows()
camera.release()