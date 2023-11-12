# Load-in model and process it
import tensorflow as tf
# model = tf.keras.models.load_model('Model_1')
model = tf.keras.models.load_model('Model_4')
# model = tf.keras.models.load_model('Model_5_BatchNorm')
# model = tf.keras.models.load_model('Model_7_Extra_Conv_TanhSigmoid')
import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
vid = cv2.VideoCapture(0)
# vid.set(cv.CAP_PROP_FPS, 1)
# print(face_cascade)
while vid.isOpened():
    ret, frame = vid.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame)
        prediction = None
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            roi_gray = gray[y:y+h, x:x+h]
            resized = cv2.resize(roi_gray, (48, 48))    # resizes to 48x48 sized image
            prediction = model.predict(np.expand_dims(np.expand_dims(resized, 0), -1))
        if prediction is not None:
          # print(prediction[0][0])
          img = cv2.resize(resized, (480, 480))
          display =""
          if np.argmax(prediction)==0:
            display ="look very angry! Im sorry!"
          elif np.argmax(prediction)==1:
            display ="are highly disgusted"
          elif np.argmax(prediction)==2:
            display ="seem afraid, what's wrong?"
          elif np.argmax(prediction)==3:
            display ="look really happy! Got some good news?"
          elif np.argmax(prediction)==4:
            display ="look quite sad, can I help cheer you up?"
          elif np.argmax(prediction)==5:
            display ="look surprised..haha!!"
          elif np.argmax(prediction)==6:
            display ="are indeed a poker face!"
          cv2.putText(frame, f"You {display}", (x-60, y-120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0),1,cv2.LINE_AA)
          cv2.putText(frame, f"Angry: {np.round(prediction[0][0],0)}", (10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
                      cv2.LINE_AA)
          cv2.putText(frame, f"Disgusted:{np.round(prediction[0][1],0)}", (10,155),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
                      cv2.LINE_AA)
          cv2.putText(frame, f"Afraid:{np.round(prediction[0][2],0)}", (10,170),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
                      cv2.LINE_AA)
          cv2.putText(frame, f"Happy:{np.round(prediction[0][3],0)}", (10,185), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
                      cv2.LINE_AA)
          cv2.putText(frame, f"Sad:{np.round(prediction[0][4],0)}",  (10,200),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
                      cv2.LINE_AA)
          cv2.putText(frame, f"Surprised:{np.round(prediction[0][5],0)}",  (10,215),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
                      cv2.LINE_AA)
          cv2.putText(frame, f"Neutral:{np.round(prediction[0][6],0)}",  (10,230),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
                      cv2.LINE_AA)

          cv2.imshow('video',frame)
        if cv2.waitKey(30) == 27:
            break
vid.release()
cv2.destroyAllWindows()

