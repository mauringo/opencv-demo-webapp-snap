from flask import Flask, redirect, render_template, request, session, url_for, Response
import cv2
from pyzbar import pyzbar
import json
import time
import os
import numpy as np

app = Flask(__name__, static_url_path='')






#cascade to identify face and eyes
face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
eyes2 = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')



print("Started")
_SETTINGS={ 
            
            'camera1': {'source': -1, 'mode':0},
            'camera2': {'source': 2, 'mode':0}
            
          }

_CAMERACHANGE = False
_CAMERA1MODE = 0
_CAMERA2MODE = 0
_STATE={'status':'starting'}


def loadSettings():
    global _SETTINGS
    try:
        with open('settings.json') as json_file: 
            _SETTINGS = json.load(json_file) 
    except IOError:
        print("File not accessible")

##########
loadSettings()
# 
camera1 = cv2.VideoCapture(_SETTINGS['camera1']['source'])  
camera2 = cv2.VideoCapture(_SETTINGS['camera2']['source']) 

_CAMERA1MODE = 3
_CAMERA2MODE = 0

def RecreateCameras():
    global camera1,camera2
    camera1 = cv2.VideoCapture(_SETTINGS['camera1']['source'])  
    camera2 = cv2.VideoCapture(_SETTINGS['camera2']['source']) 
  
# rest API PART ===========================================================================================


#save actual configuration to file
@app.route('/savesettings',methods=['POST'])
def savesettings():
    info={}
    global _SETTINGS


    with open('settings.json', 'w') as outfile:
        json.dump(_SETTINGS, outfile)

    info['status']="Setting Saved "
    resp = json.dumps(info)
    return resp
        
        

@app.route('/changemode',methods=['POST'])
def changemode():
    global _CAMERACHANGE, _CAMERA1MODE, _CAMERA2MODE
    global _SETTINGS
    info={}
    
    try:
        
       
        datar = str(request.data.decode('UTF-8'))
        print (datar)
        obj = json.loads(datar)
        cameranumber=obj['cameranumber']
        mode=obj['cameramode']
        
        info['status']="Success"
        
        if cameranumber==1:
            _SETTINGS['camera1']['mode']=mode
            _CAMERA1MODE = mode
        elif cameranumber==2:
            _SETTINGS['camera2']['mode']=mode
            _CAMERA2MODE == mode 
        else:
             info['status']="Camera Not Found"
        
        print(_CAMERA1MODE)
       
        resp = json.dumps(info)
        return resp
    except:
        info['status']="Error "
        resp = json.dumps(info)
        return resp


#save release cameras
@app.route('/releasecameras',methods=['POST'])
def releasecameras():
    global camera1,camera2
    global _CAMERACHANGE
    _CAMERACHANGE=True
    camera1.release()
    camera2.release()
    #save release cameras
    info={}

    info['status']="CamerasReleased "
    resp = json.dumps(info)
    return resp


@app.route('/restartcameras',methods=['POST'])
def restartcameras():
    global camera1,camera2
    info={}
    global _CAMERACHANGE
    camera1.release()
    camera2.release()
  
    RecreateCameras()    
    _CAMERACHANGE=False

    info['status']="Cameras re engaged, refresh page "
    resp = json.dumps(info)
    return resp



@app.route('/changecamera',methods=['POST'])
def camerachange():
    global camera1,camera2
    global _CAMERACHANGE
    global _SETTINGS
    info={}

    try:
        
        _CAMERACHANGE =True
        camera1.release()
        camera2.release()

        datar = str(request.data.decode('UTF-8'))
        print (datar)
        obj = json.loads(datar)
        cameranumber=obj['cameranumber']
        camerasource=obj['camerasource']
        
        info['status']="Success"
        
        if cameranumber==1:
            _SETTINGS['camera1']['source']=camerasource
            camera1 = cv2.VideoCapture(obj['camerasource'])
            RecreateCameras()    

        elif cameranumber==2:
            _SETTINGS['camera2']['source']=camerasource
            camera2 == cv2.VideoCapture(obj['camerasource'])  
            RecreateCameras()    

        else:
             info['status']="Camera Not Found"
        
        
        _CAMERACHANGE=False
        resp = json.dumps(info)
        return resp
    except:
        info['status']="Error "
        _CAMERACHANGE=False
        resp = json.dumps(info)
        return resp
        
#=======OPENCV FUNCTIONS :)==========================================================================================
def faceDetection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
  
    # Draw the rectangle around each face
 
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        
    return image

def barcodesDetection(image):
    barcodes = pyzbar.decode(image)
    for barcode in barcodes:
        # extract the bounding box location of the barcode and draw the
        # bounding box surrounding the barcode on the image
        (x, y, w, h) = barcode.rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # the barcode data is a bytes object so if we want to draw it on
        # our output image we need to convert it to a string first
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        # draw the barcode data and barcode type on the image
        text = "{} ({})".format(barcodeData, barcodeType)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 255), 2)

    return image

def circelesDetection(image):
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.HoughCircles function has a lot of parameters, so you can find more about it in documentation
    # or you can use cv2.HoughCircles? in jupyter nootebook to get that 

    # Check to see if there is any detection v2, 32.0, 30, 550
    #circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.6, 40)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # show the output image
    return output

def leafstatus(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # find the green color 
    mask_green = cv2.inRange(hsv, (36,0,0), (86,255,255))
    # find the brown color
    mask_brown = cv2.inRange(hsv, (8, 60, 20), (30, 255, 200))
    # find the yellow color in the leaf
    mask_yellow = cv2.inRange(hsv, (21, 39, 64), (40, 255, 255))

    mask = cv2.bitwise_or(mask_brown, mask_yellow)
    mask = cv2.bitwise_or(mask, mask_green)

    resGreen = cv2.bitwise_and(image,image, mask= mask_green)
    resBrown = cv2.bitwise_and(image,image, mask= mask_brown)
    resYellow = cv2.bitwise_and(image,image, mask= mask_yellow)
    
    yellows='Yellow: '+ str(percentage(resYellow))+'%'
    browns='Brown: '+ str(percentage(resBrown))+'%'
    greens='Green: '+ str(percentage(resGreen))+'%'
    

    resYellow=writeOnImage(resYellow,yellows)
    resBrown=writeOnImage(resBrown,browns)
    resGreen=writeOnImage(resGreen,greens)
    img=writeOnImage(image,'Normal Image')

    #create overall thing
    vis = np.concatenate((img, resBrown), axis=0)
    vis1 = np.concatenate((resGreen, resYellow), axis=0)
    vistot = np.concatenate((vis, vis1), axis=1)

    return vistot

def percentage(img):
    res1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nonZero = cv2.countNonZero(res1)
    percentage = (nonZero / (img.shape[0] * img.shape[1])) * 100
    percentage=int(percentage*10)/10
    return percentage

def writeOnImage(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # org 
    org = (65, 65) 
    # fontScale 
    fontScale = 2
    # color in BGR 
    color = (0, 255, 30) 
    # Line thickness of 2 px 
    thickness = 3
    # Using cv2.putText() method 
    img = cv2.putText(img,text, org, font,  
                    fontScale, color, thickness, cv2.LINE_AA) 

    return img

#=======Frontend =========================================================================================
def selecMode(image, cameramode):
    
    if cameramode == 0:
        return image
    elif cameramode == 1:
        return circelesDetection(image)
    elif cameramode == 2:
        return faceDetection(image)
    elif cameramode == 3:
        return barcodesDetection(image)
    elif cameramode == 4:
        return leafstatus(image)
    else:
        return image
      
      
def gen_frames1(): 
    global camera1 # generate frame by frame from camera
    global _CAMERA1MODE
    while True:

      
        if not _CAMERACHANGE:
            success, frame = camera1.read()  # read the camera frame
            if not success:
                break
            else:
                frame = selecMode(frame,_CAMERA1MODE)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed1')
def video_feed1():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames1(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames2(): 
    global camera2 # generate frame by frame from camera
    global _CAMERA2MODE
    while True:
        
        # Capture frame-by-frame
      
        if not _CAMERACHANGE:
            success, frame = camera2.read()  # read the camera frame
            if not success:
                break
            else:
                resize(frame, frame, Size(640, 360), 0, 0, INTER_CUBIC);

                frame = selecMode(frame,_CAMERA2MODE)
                ret, buffer = cv2.imencode('.jpg', frame)
            
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed2')
def video_feed2():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames2(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug = False, port=33443)