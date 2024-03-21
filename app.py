import cv2
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from flask import Flask, render_template, Response
from keras.models import load_model
from skimage.transform import resize, pyramid_reduce


app = Flask(__name__)

labels = [
"A",
"B",
"C",    
"D",
"E",
"F",
"G",
"H",
"I",
"J",
"K",
"L",
"M",
"N",
"O",
"P",
"Q",
"R",
"S",
"T",
"U",
"V",
"W",
"X",
"Y",
"Z",
]
model = load_model('model.h5')
offset=20

detector = HandDetector(maxHands=1)

def get_square(image, square_size):
    
    height, width = image.shape    
    if(height > width):
      differ = height
    else:
      differ = width
    differ += 4


    mask = np.zeros((differ, differ), dtype = "uint8")

    x_pos = int((differ - width) / 2)
    y_pos = int((differ - height) / 2)

   
    mask[y_pos: y_pos + height, x_pos: x_pos + width] = image[0: height, 0: width]

 
    if differ / square_size > 1:
      mask = pyramid_reduce(mask, differ / square_size)
    else:
      mask = cv2.resize(mask, (square_size, square_size), interpolation = cv2.INTER_AREA)
    return mask


def keras_predict(model, image):
    data = np.asarray( image, dtype="int32" )
    
    pred_probab = model.predict(data)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

def keras_process_image(img):
    
    image_x = 28
    image_y = 28
    #img = cv2.resize(img, (28,28), interpolation = cv2.INTER_AREA)
    img = get_square(img, 28)
    img = np.reshape(img, (image_x, image_y))
    
    
    return img
 

def crop_image(image, x, y, width, height):
    return image[y:y + height, x:x + width]

def generate_frames():
    while True: 
        try: 
            cam_capture = cv2.VideoCapture(0)

            _, image_frame = cam_capture.read()  

            hands, _ = detector.findHands(image_frame)
            print(hands)
            if not hands:
                # Display message or perform an alternative action (e.g., continue)
                cv2.putText(image_frame, "No hand detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                _, jpeg = cv2.imencode('.jpg', image_frame)
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                continue

            hand = hands[0]
            print(hand['lmList'])
            x1, y1 = hand['lmList'][0][0], hand['lmList'][0][1]# Wrist coordinates (adjust as needed)
            x2, y2 = hand['lmList'][5][0], hand['lmList'][5][1]  # Middle fingertip coordinates (adjust as needed)
            x3, y3 = hand['lmList'][4][0], hand['lmList'][4][1]  # Thumb fingertip coordinates (adjust as needed)
            x4, y4 = hand['lmList'][20][0], hand['lmList'][20][1]  # tiny fingertip coordinates (adjust as needed)

            # Calculate ROI dimensions
            width = abs(x3 - x4)+160
            height = abs(y2 - y1)+400
            
            # Select ROI
            im2 = crop_image(image_frame, min(x3,x4)-80, y2, width, height)
            image_grayscale = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
            image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (15, 15), 0)

            # Resize the frame for display
            display_img = cv2.resize(image_frame, (640,480))

            # Perform prediction on the original resized image
            resized_img = cv2.resize(image_grayscale_blurred, (28, 28))
            ar = resized_img.reshape(1, 784)

            pred_probab, pred_class = keras_predict(model, ar)
            print("Predicted Class" , pred_class)
            predicted_label = labels[pred_class]

            # Draw the predicted label on the resized frame
            cv2.putText(display_img, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Encode the frame as JPEG
            _, jpeg = cv2.imencode('.jpg', display_img)

            # Convert the frame to bytes
            frame_bytes = jpeg.tobytes()

            # Yield frame for video streaming
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Wait for 'q' key to stop the streaming
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            print("Error")
        
    cam_capture.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)