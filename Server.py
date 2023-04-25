import cv2  # for web camera
import os
# from scipy.misc import imread
from flask import Flask, request, render_template
import numpy as np
import base64
import io
from imageio import imread

from utils import getMaskedResult,Detect_label
import time
app = Flask(__name__, template_folder='templates', static_folder='static')

dir_path = os.path.dirname(os.path.realpath(__file__))
IMG_FOLDER = os.path.join('static', 'images')
app.config['IMG_FOLDER'] = IMG_FOLDER

app.secret_key = os.urandom(24)

# API functions
@app.route("/")
def index_page():
    """Renders the 'index.html' page for manual image file uploads."""
    full_filename = os.path.join(app.config['IMG_FOLDER'], 'avatar.png')
    return render_template("main.html")

@app.route('/COLOR_Post',  methods=['POST'])
def colorBkground():
    colorvalue = request.values["color"]
    print(colorvalue)
    colorvalue_sole=request.values["color_sole"]
    print(colorvalue_sole)
    image_origin=cv2.imread("networks/img_tmp.png")
    image_mask=cv2.imread("networks/mask_tmp.png",0)
    image_mask_sole=cv2.imread("networks/mask_sole_tmp.png",0)
    imgresult = getMaskedResult(image_origin, image_mask,image_mask_sole, colorvalue,colorvalue_sole)
    imgresult=imageprocessmargin(imgresult)
    _, img_arr = cv2.imencode('.png', imgresult)
    img_base64 = base64.b64encode(img_arr.tobytes()).decode('utf-8')
    img_base64 = "data:image/png;base64," + img_base64
    # print(img_base64)
    return img_base64

@app.route('/imgPortrait',  methods=['POST'])
def Loadimage():
    st=time.time()
    img_original = request.values["portraitphoto"]

    base64str = img_original
    if len(base64str) > 0:
        imgtmp = imread(io.BytesIO(base64.b64decode(base64str)))
        img_portrait = cv2.cvtColor(imgtmp, cv2.COLOR_RGB2BGR)
    else:
        return ""
    if img_portrait is None:
        return ""
    imgresult=imageprocessmargin(img_portrait)

    _, img_arr = cv2.imencode('.png', imgresult)
    img_base64 = base64.b64encode(img_arr.tobytes()).decode('utf-8')
    img_base64 = "data:image/png;base64," + img_base64
    # print(img_base64)
    et=time.time()
    print(et-st,"loading")
    return img_base64

@app.route('/RmbkPortrait',  methods=['POST'])
def removeBkground():

    img_original = request.values["portraitphoto"]
    colorvalue = request.values["color"]
    colorvalue_sole=request.values["color_sole"]
    print(colorvalue)
    print(colorvalue_sole)
    base64str = img_original
    if len(base64str) > 0:
        imgtmp = imread(io.BytesIO(base64.b64decode(base64str)))
        img_portrait = cv2.cvtColor(imgtmp, cv2.COLOR_RGB2BGR)
    else:
        return ""
    if img_portrait is None:
        return ""

    img_mask,img_mask_sole=Detect_label(img_portrait)
    imgresult = getMaskedResult(img_portrait,img_mask,img_mask_sole,colorvalue,colorvalue_sole)
    imgresult=imageprocessmargin(imgresult)
    _, img_arr = cv2.imencode('.png', imgresult)
    img_base64 = base64.b64encode(img_arr.tobytes()).decode('utf-8')
    img_base64 = "data:image/png;base64," + img_base64
    # print(img_base64)
    return img_base64

def imageprocessmargin(image):
    st=time.time()
    h,w,_=image.shape
    ratio=w/h
    size=2500
    tmp=cv2.resize(image,(size,size))
    if ratio>1:
        tmp_h=int(h * size / w)
        diff=size-tmp_h
        image = cv2.resize(image, (size,tmp_h))
        tmp[0:int(diff/2),:]=255
        tmp[int(diff/2):int(diff/2)+tmp_h,:]=image
        tmp[int(diff/2)+tmp_h:size,:]=255
    else:
        tmp_w=int(w * size / h)
        diff=size-tmp_w
        image = cv2.resize(image, (tmp_w,size))
        tmp[:,0:int(diff/2)]=255
        tmp[:,int(diff/2):int(diff/2)+tmp_w]=image
        tmp[:,int(diff/2)+tmp_w:size]=255
    et=time.time()
    print(et-st,"processing",size)
    return tmp


if __name__ == "__main__":

    # Start flask application on waitress WSGI server
    app.run(host='0.0.0.0', port=5000)