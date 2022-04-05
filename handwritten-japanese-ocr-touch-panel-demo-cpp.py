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

from text_detection_postprocess import postprocess

from openvino.preprocess import PrePostProcessor, ResizeAlgorithm
from openvino.runtime import AsyncInferQueue, Core, InferRequest, Layout, Type
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

g_lnk_th = 50
g_cls_th = 15
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
    global g_recogFlag
    col = (0,255,255) if not g_recogFlag else (0,128,128)
    cv2.circle(image, (0               , 0), 100, col, -1)
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

def onTrackbarLnk(x):
    global g_lnk_th
    g_lnk_th = x

def onTrackbarCls(x):
    global g_cls_th
    g_cls_th = x

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
    core = Core()

    model_root = '.'

    # text-detection-0003  in: (1,768,1280,3)  out: model/link_logits_/add(1,192,320,16) model/segm_logits/add(1,192,320,2)
    model='text-detection-0003'
    model = os.path.join(model_root, 'intel', model, 'FP16', model)
    net_td = core.read_model(model+'.xml')
    ppp = PrePostProcessor(net_td)
    ppp.input().tensor().set_element_type(Type.u8).set_layout(Layout('NHWC'))
    ppp.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)
    net_td = ppp.build()
    compiled_model_td = core.compile_model(net_td, 'CPU')
    ireq_td = compiled_model_td.create_infer_request()

    # handwritten-japanese-recognition
    model = 'handwritten-japanese-recognition-0001'
    model = os.path.join(model_root, 'intel', model, 'FP16', model)
    net = core.read_model(model+'.xml')
    input_batch_size, input_channel, input_height, input_width = list(net.input(0).get_shape())
    compiled_model = core.compile_model(net, 'CPU')
    ireq = compiled_model.create_infer_request()

    characters = get_characters('data/kondate_nakayosi_char_list.txt')
    codec = CTCCodec(characters)

    clearCanvas()
    cv2.namedWindow('canvas')
    cv2.setMouseCallback('canvas', onMouse)
    cv2.createTrackbar('Link           Threshold', 'canvas', 50, 100, onTrackbarLnk)
    cv2.createTrackbar('Classification Threshold', 'canvas', 15, 100, onTrackbarCls)

    while True:
        g_UIState = 0
        while g_recogFlag==False:
            key=cv2.waitKey(100)
            dispCanvas()
            if key==27:
                return
            if key==ord(' '):
                break
        cv2.waitKey(1)
        g_recogFlag = False
        g_UIState = 1

        print('text detection')
        tensor = np.expand_dims(g_canvas, 0)
        res_td = ireq_td.infer({0: tensor})
        # To access to the inference result, either one of following way is OK.
        link = ireq_td.get_tensor('model/link_logits_/add:0').data   # 'model/link_logits_/add'  1,192,320,16
        segm = ireq_td.get_tensor('model/segm_logits/add:0').data    # 'model/segm_logits/add'   1,192,320,2
        #link = ireq_td.get_tensor(compiled_model_td.output(1)).data   # 'model/link_logits_/add'  1,192,320,16
        #segm = ireq_td.get_tensor(compiled_model_td.output(0)).data    # 'model/segm_logits/add'   1,192,320,2
        rects = postprocess(link, segm, _canvas_x, _canvas_y, g_lnk_th/100., g_cls_th/100.)

        canvas2 = g_canvas.copy()
        for i, rect_ in enumerate(rects):
            rect = ((rect_[0], rect_[1]), (rect_[2], rect_[3]), rect_[4])
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.polylines(canvas2, [box], True, (255,0,0), 4)

            most_left_idx, most_left = topLeftPoint(box)
            crop = cropRotatedImage(g_canvas, box, most_left_idx)
            input_image = preprocess_input(crop, input_height, input_width)[None,:,:,:]

            res = ireq.infer({0: input_image})
            preds = ireq.get_tensor(compiled_model.output(0)).data
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
