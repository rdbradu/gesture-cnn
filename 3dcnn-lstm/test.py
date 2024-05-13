import tensorflow as tf
import numpy as np
import cv2
from process import rgb2gray, normalize_data

classes = ['Swiping Up', 'Swiping Right', 'Swiping Left', 'Doing other things']

def main():
    model = tf.keras.models.load_model("3dcnn_lstm.h5")
    model.summary()

    to_predict = []
    num_frames = 0
    cap = cv2.VideoCapture(0)
    classe =''

    while(True):
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        to_predict.append(cv2.resize(gray, (64, 64)))
            
        if len(to_predict) == 15:
            frame_to_predict = np.array(to_predict, dtype=np.float32)
            frame_to_predict = normalize_data(frame_to_predict)
            predict = model.predict(frame_to_predict)

            try:
                current = classes[np.argmax(predict)]            

                if np.amax(predict) * 100 >= 50:
                    print('Class = ',current, 'Precision = ', np.amax(predict)*100,'%')
            except:
                pass

            to_predict = []
        cv2.putText(frame, classe, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),1,cv2.LINE_AA)


        # # Display the resulting frame
        cv2.imshow('Hand Gesture Recognition',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
main()