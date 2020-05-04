"""
Handwritten Japanese OCR demo program
 Based on a sample program from OpenVINO 2020.2 (handwritten-japanese-recognition-demo.py)
"""

"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import sys
import time
import math
import logging as log
from argparse import ArgumentParser, SUPPRESS

import cv2
import numpy as np
from functools import reduce

from PIL import ImageFont, ImageDraw, Image

from openvino.inference_engine import IENetwork, IECore
from utils.codec import CTCCodec

# Canvas size is the same as the input size of the text detection model (to ommit resizing before text area inference)
_canvas_x = 1280
_canvas_y = 768


# -----------------------------------------------------------------

def get_characters(char_file):
    with open(char_file, 'r', encoding='utf-8') as f:
        return ''.join(line.strip('\n') for line in f)


def preprocess_input(src, height, width):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ratio = float(src.shape[1]) / float(src.shape[0])
    tw = int(height * ratio)

    rsz = cv2.resize(src, (tw, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    outimg = np.full((height, width), 255., np.float32)
    rsz_h, rsz_w = rsz.shape
    outimg[:rsz_h, :rsz_w] = rsz
    cv2.imshow('OCR input image', outimg)

    outimg = np.reshape(outimg, (1, height, width))
    return outimg

# -----------------------------------------------------------------

def softmax_channel(data):
    for i in range(0, len(data), 2):
        m=max(data[i], data[i+1])
        data[i  ] = math.exp(data[i  ]-m)
        data[i+1] = math.exp(data[i+1]-m)
        s=data[i  ]+data[i+1]
        data[i  ]/=s
        data[i+1]/=s
    return data


def findRoot(point, group_mask):
    root = point
    update_parent = False
    while group_mask[root] != -1:
        root = group_mask[root]
        update_parent = True
    if update_parent:
        group_mask[point] = root
    return root


def join(p1, p2, group_mask):
    root1 = findRoot(p1, group_mask)
    root2 = findRoot(p2, group_mask)
    if root1 != root2:
        group_mask[root1] = root2


def get_all(points, w, h, group_mask):
    root_map = {}
    mask = np.zeros((h, w), np.int32)
    for px, py in points:
        point_root = findRoot(px+py*w, group_mask)
        if not point_root in root_map:
            root_map[point_root] = len(root_map)+1
        mask[py, px] = root_map[point_root]
    return mask


def decodeImageByJoin(segm_data, segm_data_shape, link_data, link_data_shape, segm_conf_thresh, link_conf_thresh):
    h = segm_data_shape[1]
    w = segm_data_shape[2]
    pixel_mask = np.full((h*w,), False, dtype=np.bool)
    group_mask = {}
    points     = []
    for i, segm in enumerate(segm_data):
        if segm>segm_conf_thresh:
            pixel_mask[i] = True
            points.append((i%w, i//w))
            group_mask[i] = -1
        else:
            pixel_mask[i] = False
    
    link_mask = np.array([ ld>=link_conf_thresh for ld in link_data ])

    neighbours = int(link_data_shape[3])
    for px, py in points:
        neighbor = 0
        for ny in range(py-1, py+1+1):
            for nx in range(px-1, px+1+1):
                if nx==px and ny==py:
                    continue
                if nx<0 or nx>=w or ny<0 or ny>=h:
                    continue
                pixel_value = pixel_mask[ny*w + nx]
                link_value  = link_mask [py*w + px*neighbours + neighbor ]
                if pixel_value and link_value:
                    join(px+py*w, nx+ny*w, group_mask)
                neighbor+=1
    return get_all(points, w, h, group_mask)


def maskToBoxes(mask, min_area, min_height, image_size):
    _X=0
    _Y=1
    bboxes = []
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(mask)
    max_bbox_idx = int(max_val)
    resized_mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)

    for i in range(1, max_bbox_idx+1):
        bbox_mask = np.where(resized_mask==i, 255, 0).astype(np.uint8)
        contours, hierarchy = cv2.findContours(bbox_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)==0:
            continue
        center, size, angle = cv2.minAreaRect(contours[0])
        if min(size[_X], size[_Y]) < min_height:
            continue
        if size[_X]*size[_Y] < min_area:
            continue
        bboxes.append((center, size, angle))
    return bboxes


def text_detection_postprocess(link, segm, image_size, segm_conf_thresh, link_conf_thresh):
    _N = 0
    _C = 1
    _H = 2
    _W = 3
    kMinArea   = 300
    kMinHeight = 10

    link_shape = link.shape
    link_data_size = reduce(lambda a, b: a*b, link_shape)
    link_data = link.transpose((_N, _H, _W, _C))
    link_data = link_data.flatten()
    link_data = softmax_channel(link_data)
    link_data = link_data.reshape((-1,2))[:,1]
    new_link_data_shape = [ link_shape[0], link_shape[2], link_shape[3], link_shape[1]/2 ]

    segm_shape = segm.shape
    segm_data_size = reduce(lambda a, b: a*b, segm_shape)
    segm_data = segm.transpose((_N, _H, _W, _C))
    segm_data = segm_data.flatten()
    segm_data = softmax_channel(segm_data)
    segm_data = segm_data.reshape((-1,2))[:,1]
    new_segm_data_shape = [ segm_shape[0], segm_shape[2], segm_shape[3], segm_shape[1]/2 ]

    mask = decodeImageByJoin(segm_data, new_segm_data_shape, link_data, new_link_data_shape, 
                             segm_conf_thresh, link_conf_thresh)
    rects = maskToBoxes(mask, kMinArea, kMinHeight, image_size)

    return rects



# ----------------------------------------------------------------------------

def topLeftPoint(points):
    big_number = 1e10
    _X=0
    _Y=1
    most_left        = [big_number, big_number]
    almost_most_left = [big_number, big_number]
    most_left_idx        = -1
    almost_most_left_idx = -1

    for i, point in enumerate(points):
        px, py = point
        if most_left[_X]>px:
            if most_left[_X]<big_number:
                almost_most_left     = most_left
                almost_most_left_idx = most_left_idx
            most_left = [px, py]
            most_left_idx = i
        if almost_most_left[_X] > px and [px,py]!=most_left:
            almost_most_left = [px,py]
            almost_most_left_idx = i
    if almost_most_left[_Y]<most_left[_Y]:
        most_left     = almost_most_left
        most_left_idx = almost_most_left_idx
    return most_left_idx, most_left


def cropRotatedImage(image, points, top_left_point_idx):
    _X=1
    _Y=0
    _C=2
    
    point0 = points[ top_left_point_idx       ]
    point1 = points[(top_left_point_idx+1) % 4]
    point2 = points[(top_left_point_idx+2) % 4]
    
    target_size = (int(np.linalg.norm(point2-point1, ord=2)), int(np.linalg.norm(point1-point0, ord=2)), 3)

    crop = np.full(target_size, 255, np.uint8)
    
    _from = np.array([ point0, point1, point2 ], dtype=np.float32)
    _to   = np.array([ [0,0], [target_size[_X]-1, 0], [target_size[_X]-1, target_size[_Y]-1] ], dtype=np.float32)

    M    = cv2.getAffineTransform(_from, _to)
    crop = cv2.warpAffine(image, M, (target_size[_X], target_size[_Y]))

    return crop

# ----------------------------------------------------------------------------

g_mouseX=-1
g_mouseY=-1
g_mouseBtn = -1  # 0=left, 1=right, -1=none

g_UIState = 0       # 0: normal UI, 1: wait for a click
g_clickedFlag = False
g_recogFlag   = False

g_threshold = 50
g_canvas = []

def putJapaneseText(img, x, y, text, size=32):
    if os.name =='nt':
        #fontName = 'meiryo.ttc'                 # Win10
        fontName = 'msgothic.ttc'                 # Win10
    elif os.name == 'posix':
        fontName = 'NotoSansCJK-Regular.ttc'    # Ubuntu
    elif os.name == 'Darwin':
        fontName = 'Osaka.ttf'                  # Not tested ...
    else:
        fontName = 'UnknownOS'

    try:
        font = ImageFont.truetype(fontName, size)
    except IOError:
        cv2.putText(img, 'font "{}" not found'.format(fontName), (x,y-8), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
    else:
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        w,h = draw.textsize(text, font)
        draw.text((x, y-h*1.2), text, font=font, fill=(255,0,0,0))
        img = np.array(img_pil)

    return img


def drawUI(image):
    cv2.circle(image, (0               , 0), 100, (   0, 255, 255), -1)
    cv2.circle(image, (image.shape[1]-1, 0), 100, (   0, 255,   0), -1)
    cv2.putText(image, 'RECOGNIZE', (4                 ,20), cv2.FONT_HERSHEY_PLAIN, 1, (  0,   0,   0), 2)
    cv2.putText(image, 'CLEAR'    , (image.shape[1]-60 ,20), cv2.FONT_HERSHEY_PLAIN, 1, (  0,   0,   0), 2)


def clearCanvas():
    global g_canvas
    g_canvas = np.full((_canvas_y, _canvas_x, 3), [255,255,255], np.uint8)


def dispCanvas():
    global g_canvas
    canvas = g_canvas.copy()
    drawUI(canvas)
    cv2.imshow('canvas', canvas)
    cv2.waitKey(1)


# Mouse event handler
def onMouse(event, x, y, flags, param):
    global g_mouseX, g_mouseY
    global g_mouseBtn
    global g_recogFlag
    global g_clickedFlag
    global g_UIState

    global g_canvas

    black_pen = lambda x1, y1, x2, y2: cv2.line(g_canvas, (x1, y1), (x2, y2), (  0,  0,  0), thickness=12)
    white_pen = lambda x1, y1, x2, y2: cv2.line(g_canvas, (x1, y1), (x2, y2), (255,255,255), thickness=36)

    if g_UIState==0:      # Normal UI
        if event == cv2.EVENT_LBUTTONDOWN:
            p0=np.array([        0, 0])
            p1=np.array([_canvas_x, 0])
            pp=np.array([        x, y])
            if np.linalg.norm(pp-p0, ord=2)<100:        # Recognition
                g_recogFlag = True
            elif np.linalg.norm(pp-p1, ord=2)<100:      # Clear
                clearCanvas()
            else:
                g_mouseBtn = 0      # left button
        if event == cv2.EVENT_LBUTTONUP:
            if g_mouseBtn==0:
                black_pen(g_mouseX, g_mouseY, x, y)
            g_mouseBtn = -1
        if event == cv2.EVENT_RBUTTONDOWN:
            g_mouseBtn = 1          # right button
        if event == cv2.EVENT_RBUTTONUP:
            if g_mouseBtn==1:
                white_pen(g_mouseX, g_mouseY, x, y)
            g_mouseBtn = -1
        if event == cv2.EVENT_MOUSEMOVE:
            if   g_mouseBtn==0:
                black_pen(g_mouseX, g_mouseY, x, y)
            elif g_mouseBtn==1:
                white_pen(g_mouseX, g_mouseY, x, y)
    elif g_UIState==1:      # no draw. wait for click state
        if event == cv2.EVENT_LBUTTONUP:
            g_clickedFlag=True

    g_mouseX = x
    g_mouseY = y

def onTrackbar(x):
    global g_threshold
    g_threshold = x

# ----------------------------------------------------------------------------

def main():
    _H=0
    _W=1
    _C=2

    global g_canvas
    global g_threshold
    global g_UIState
    global g_recogFlag
    global g_clickedFlag

    # Plugin initialization
    ie = IECore()

    # text-detection-0003  in: (1,3,768,1280)  out: model/link_logits_/add(1,16,192,320) model/segm_logits/add(1,2,192,320)
    model='text-detection-0003'
    model = './intel/'+model+'/FP16/'+model
    net_td = ie.read_network(model+'.xml', model+'.bin')
    input_blob_td = next(iter(net_td.inputs))
    out_blob_td   = next(iter(net_td.outputs))
    exec_net_td = ie.load_network(net_td, 'CPU')

    # handwritten-japanese-recognition
    model = 'handwritten-japanese-recognition-0001'
    model = './intel/'+model+'/FP16/'+model
    net = ie.read_network(model+'.xml', model+'.bin')
    input_blob = next(iter(net.inputs))
    out_blob   = next(iter(net.outputs))
    input_batch_size, input_channel, input_height, input_width= net.inputs[input_blob].shape
    exec_net = ie.load_network(net, 'CPU')

    characters = get_characters('data/kondate_nakayosi_char_list.txt')
    codec = CTCCodec(characters)

    clearCanvas()
    cv2.namedWindow('canvas')
    cv2.setMouseCallback('canvas', onMouse)
    cv2.createTrackbar('Threshold', 'canvas', 50, 100, onTrackbar)

    while True:
        g_UIState = 0
        while g_recogFlag==False:
            dispCanvas()
            key=cv2.waitKey(100)
            if key==27:
                return
            if key==ord(' '):
                break
        g_recogFlag = False
        g_UIState = 1

        print('text detection')
        img = cv2.resize(g_canvas, (_canvas_x, _canvas_y))
        img = img.transpose((_C, _H, _W))
        img = img.reshape((1, 3, _canvas_y, _canvas_x))
        res_td = exec_net_td.infer(inputs={input_blob_td: img})
        link = res_td['model/link_logits_/add']     # 1,16,192,320
        segm = res_td['model/segm_logits/add' ]     # 1, 2,192,320
        rects = text_detection_postprocess(link, segm, (_canvas_x, _canvas_y), g_threshold/100., g_threshold/100.)
        print('text detection - completed')

        canvas2 = g_canvas.copy()
        for i, rect in enumerate(rects):
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.polylines(canvas2, [box], True, (255,0,0), 4)

            most_left_idx, most_left = topLeftPoint(box)
            crop = cropRotatedImage(g_canvas, box, most_left_idx)
            input_image = preprocess_input(crop, input_height, input_width)[None,:,:,:]

            preds = exec_net.infer(inputs={input_blob: input_image})
            preds = preds[out_blob]
            result = codec.decode(preds)
            print('OCR result ({}): {}'.format(i, result))
            
            canvas2 = putJapaneseText(canvas2, most_left[0], most_left[1], result[0])
            cv2.imshow('canvas', canvas2)
            cv2.waitKey(1)

        cv2.putText(canvas2, 'Hit any key, tap screen or click L-button to continue', (0, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
        cv2.imshow('canvas', canvas2)
        g_clickedFlag=False
        key=-1
        while g_clickedFlag==False and key==-1:
            key=cv2.waitKey(100)

    return

if __name__ == '__main__':
    print('Handwritten Japanese OCR Demo')
    print('ESC: Quit')
    print('Mouse L-Button: Draw')
    print('Mouse R-Button: Erase')
    print('Threshold = Text area detect threshold')
    main()
