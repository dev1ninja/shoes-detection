from __future__ import print_function

# import tensorflow as tf
import tensorflow.compat.v1 as tf
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


def process(origin_image,mask_image,maskcolor,mask_sole_image,masksolecolor):
    size_w=2500
    origin_image=cv2.resize(origin_image,(size_w,int(origin_image.shape[0]*size_w/origin_image.shape[1])))
    mask_image=cv2.resize(mask_image,(size_w,int(origin_image.shape[0]*size_w/origin_image.shape[1])))
    mask_sole_image=cv2.resize(mask_sole_image,(size_w,int(origin_image.shape[0]*size_w/origin_image.shape[1])))
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
    saver.restore(sess, './model_sole/segmentation.ckpt-27370')
    img=cv2.resize(image,(256,256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    # print(mask_color)
    images = []
    img = im_pil.copy()
    w, h = img.size
    # print("Image Size:" + str(w) + "_" + str(h))
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
    saver.restore(sess, './model_top/segmentation.ckpt-21000')
    img=cv2.resize(image,(256,256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    # print(mask_color)
    images = []
    img = im_pil.copy()
    w, h = img.size
    # print("Image Size:" + str(w) + "_" + str(h))
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
    cv2.imwrite("networks/result.png",imageresult)
    return imageresult




number_weights_path = "./model_detect/detectshoe.weights"
number_cfg_path = "./model_detect/detectshoe.cfg"
number_classes_path = "./model_detect/detectshoe.names"

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def Detect_label(image):
    image_mask_sole=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_mask_top=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_mask_sole[:,:]=0
    image_mask_top[:,:]=0

    number_net = cv2.dnn.readNet(number_weights_path, number_cfg_path)
    origin=image.copy()
    Width = image.shape[1]
    Height = image.shape[0]

    scale = 0.00392
    size_w = 1024
    image = cv2.resize(image, (size_w, int(Height * size_w / Width)))


    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    number_net.setInput(blob)

    outs = number_net.forward(get_output_layers(number_net))

    class_ids = []
    confidences = []
    boxes = []
    center_X = []
    center_Y = []
    points = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                point = (x, y)
                center_X.append(center_x)
                center_Y.append(center_y)
                points.append(point)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    valid_boxes = []
    valid_classids = []
    valid_centerX = []


    t = 0
    for i in indices:
        box = boxes[t]
        x = box[0]
        valid_boxes.append(box)
        valid_classids.append(class_ids[t])
        valid_centerX.append(x)
        t += 1

    for i in range(0, len(valid_classids)):
        box = valid_boxes[i]
        x = int(box[0])-15
        y = int(box[1])-15
        w = int(box[2])+30
        h = int(box[3])+30
        shoe_part=origin[y:y+h,x:x+w]
        sh,sw,_=shoe_part.shape
        mask_sole=getMaskSoleImage(shoe_part)
        mask_top=getMaskImage(shoe_part)
        mask_sole=cv2.resize(mask_sole,(sw,sh))
        mask_top=cv2.resize(mask_top,(sw,sh))


        make_maskimage(mask_top,x,y,sw,sh,image_mask_top)
        make_maskimage(mask_sole,x,y,sw,sh,image_mask_sole)
    cv2.imwrite("networks/img_tmp.png",origin)
    cv2.imwrite("networks/mask_tmp.png",image_mask_top)
    cv2.imwrite("networks/mask_sole_tmp.png",image_mask_sole)
    return image_mask_top,image_mask_sole
def make_maskimage(mask,x,y,sw,sh,whole_mask):
    whole_mask[y:y+sh,x:x+sw]=mask


