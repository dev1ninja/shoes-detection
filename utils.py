from __future__ import print_function

import tensorflow as tf
import numpy as np
from PIL import Image
from PIL import ImageColor

import cv2
import networks.unet

x = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='x')
y_pred = networks.unet.create_unet(x, train=False)
y_pred = tf.argmax(y_pred, axis=3, name="y_pred")

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
# saver.restore(sess, './model_sole/segmentation.ckpt-7422')

def process(origin_image,mask_image,maskcolor,mask_sole_image,masksolecolor):
    origin_image=cv2.resize(origin_image,(512,int(origin_image.shape[0]*512/origin_image.shape[1])))
    mask_image=cv2.resize(mask_image,(512,int(origin_image.shape[0]*512/origin_image.shape[1])))
    mask_sole_image=cv2.resize(mask_sole_image,(512,int(origin_image.shape[0]*512/origin_image.shape[1])))
    mask_color = ImageColor.getcolor(maskcolor, "RGB")
    print(mask_color)
    mask_color = np.array(mask_color, dtype=np.float)
    mask_color = np.flip(mask_color)
    mask_color /= 255.0
    mask_sole_color = ImageColor.getcolor(masksolecolor, "RGB")
    print(mask_sole_color)
    mask_sole_color = np.array(mask_sole_color, dtype=np.float)
    mask_sole_color = np.flip(mask_sole_color)
    mask_sole_color /= 255.0

    # origin_image=cv2.cvtColor(origin_image,cv2.COLOR_RGB2BGR)
    origin_image = np.array(origin_image, dtype=np.float)
    origin_image /= 255.0
    mask_image=cv2.cvtColor(mask_image,cv2.COLOR_GRAY2BGR)
    mask = np.array(mask_image, dtype=np.float)
    mask /= 255.0

    mask_sole_image=cv2.cvtColor(mask_sole_image,cv2.COLOR_GRAY2BGR)
    mask_sole = np.array(mask_sole_image, dtype=np.float)
    mask_sole /= 255.0
    # set transparency to 25%
    transparency = .6
    mask *= transparency
    mask_sole*=transparency

    mask_part = np.ones(origin_image.shape, dtype=np.float) * mask_color
    mask_sole_part = np.ones(origin_image.shape, dtype=np.float) * mask_sole_color

    out = mask_part*mask + origin_image * (1.0 - mask)
    out = mask_sole_part*mask_sole + out * (1.0 - mask_sole)
    result = 255 * out  # Now scale by 255
    result = result.astype(np.uint8)


    return result

def getMaskSoleImage(image):
    saver.restore(sess, './model_sole/segmentation.ckpt-7422')
    img=cv2.resize(image,(256,256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    # print(mask_color)
    images = []
    img = im_pil.copy()
    w, h = img.size
    print("Image Size:" + str(w) + "_" + str(h))
    img = np.array(img)
    img = img.astype(np.float32)
    img = np.multiply(img, 1.0 / 255.0)
    images.append(img)
    feed_dict_tr = {x: images}
    results = sess.run(y_pred, feed_dict=feed_dict_tr)
    results.astype(np.uint8)
    results *= 255  # for visualization
    imgR = results[0, :, :]
    imgR = imgR.astype(np.uint8)
    imgR = Image.fromarray(imgR)
    pil_image = imgR
    mask_image = np.array(pil_image)
    mask_image = cv2.resize(mask_image, (w, h), interpolation=cv2.INTER_LINEAR)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    # for i in range(0, 1):
    #     mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_DILATE, kernel)
    # for i in range(0, 1):
    #     mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_ERODE, kernel)
    return mask_image


def getMaskImage(image):
    saver.restore(sess, './model/segmentation.ckpt-13930')
    img=cv2.resize(image,(256,256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    # print(mask_color)
    images = []
    img = im_pil.copy()
    w, h = img.size
    print("Image Size:" + str(w) + "_" + str(h))
    img = np.array(img)
    img = img.astype(np.float32)
    img = np.multiply(img, 1.0 / 255.0)
    images.append(img)
    feed_dict_tr = {x: images}
    results = sess.run(y_pred, feed_dict=feed_dict_tr)
    results.astype(np.uint8)
    results *= 255  # for visualization
    imgR = results[0, :, :]
    imgR = imgR.astype(np.uint8)
    imgR = Image.fromarray(imgR)
    pil_image = imgR
    mask_image = np.array(pil_image)
    mask_image = cv2.resize(mask_image, (w, h), interpolation=cv2.INTER_LINEAR)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    # for i in range(0, 1):
    #     mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_DILATE, kernel)
    # for i in range(0, 1):
    #     mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_ERODE, kernel)
    return mask_image

def getMaskedResult(imgframe,imgmask,imgmask_sole,maskcolor,mask_solecolor):
    if imgframe is None:
        return None
    if len(imgframe.shape) < 3:
        imgframe = cv2.cvtColor(imgframe, cv2.COLOR_GRAY2BGR)
    imageresult = process(imgframe,imgmask,maskcolor,imgmask_sole,mask_solecolor)
    return imageresult

