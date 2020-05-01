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

from __future__ import print_function
import os
import sys
import time
import math
import logging as log
from argparse import ArgumentParser, SUPPRESS

import cv2
import numpy as np
from functools import reduce

from openvino.inference_engine import IENetwork, IECore
from utils.codec import CTCCodec


def get_characters(char_file):
    '''Get characters'''
    with open(char_file, 'r', encoding='utf-8') as f:
        return ''.join(line.strip('\n') for line in f)

def preprocess_input(src, height, width):

    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ratio = float(src.shape[1]) / float(src.shape[0])
    tw = int(height * ratio)

    #cv2.imshow('input image', src)
    #cv2.waitKey(0)
    #src = cv2.GaussianBlur(src, (3,3), 0)
    #src = cv2.medianBlur(src, 3)
    #cv2.imshow('input image', src)
    #cv2.waitKey(0)

    #src = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
    #_, src = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)
    #kernel = np.ones((3,3), np.uint8)
    #src = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel)

    rsz = cv2.resize(src, (tw, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    outimg = np.full((height, width), 255., np.float32)
    rsz_h, rsz_w = rsz.shape
    outimg[:rsz_h, :rsz_w] = rsz
    cv2.imshow('input image', outimg)
    cv2.waitKey(2*1000)

    outimg = np.reshape(outimg, (1, height, width))
    return outimg

    '''
    rsz = cv2.resize(src, (tw, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    # [h,w] -> [c,h,w]
    img = rsz[None, :, :]
    _, h, w = img.shape
    # right edge padding
    pad_img = np.pad(img, ((0, 0), (0, height - h), (0, width -  w)), mode='edge')
    return pad_img
    '''


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

# w=320, h=192
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
    h, w = segm_data_shape[1:2+1]   # 192, 320
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
    bboxes = []
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(mask)
    max_bbox_idx = int(max_val)
    resized_mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)

    for i in range(1, max_bbox_idx+1):
        bbox_mask = np.where(resized_mask==i, 255, 0).astype(np.uint8)
        contours, hierachy = cv2.findContours(bbox_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)==0:
            continue
        center, size, angle = cv2.minAreaRect(contours[0])
        if min(size[0], size[1]) < min_height:
            continue
        if size[0]*size[1] < min_area:
            continue
        bboxes.append((center, size, angle))
    return bboxes


def text_detection_postprocess(link, segm, image_size, segm_conf_thresh, link_conf_thresh):
    #  model/link_logits_/add   (1, 16, 192, 320)
    #  model/segm_logits/add,   (1,  2, 192, 320)

    kMinArea   = 300
    kMinHeight = 10

    link_shape = link.shape
    link_data_size = reduce(lambda a, b: a*b, link_shape)
    link_data = link.transpose((0,2,3,1))   # 1,192,320,16
    link_data = link_data.flatten()
    link_data = softmax_channel(link_data)
    link_data = link_data.reshape((-1,2))[:,1]
    new_link_data_shape = [ link_shape[0], link_shape[2], link_shape[3], link_shape[1]/2 ]

    segm_shape = segm.shape
    segm_data_size = reduce(lambda a, b: a*b, segm_shape)
    segm_data = segm.transpose((0,2,3,1))   # 1,192,320,2
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
    target_size = (int(np.linalg.norm(point2-point1)), int(np.linalg.norm(point1-point0)), 3)

    crop = np.zeros(target_size, np.uint8)
    _from = np.array([ point0, point1, point2 ], dtype=np.float32)
    _to   = np.array([ [0,0], [target_size[_X]-1, 0], [target_size[_X]-1, target_size[_Y]-1] ], dtype=np.float32)
    M    = cv2.getAffineTransform(_from, _to)
    crop = cv2.warpAffine(image, M, (target_size[_X], target_size[_Y]))

    return crop

# ----------------------------------------------------------------------------


mouseX=-1
mouseY=-1

recogFlag = False

# Mouse event handler
def onMouse(event, x, y, flags, param):
    global mouseX, mouseY
    global canvas
    global recogFlag
    
    thinkness = 12

    if event == cv2.EVENT_LBUTTONDOWN:
        p0=np.array([0   ,0])
        p1=np.array([1280,0])
        pp=np.array([   x,y])
        if np.linalg.norm(pp-p0, ord=2)<100:        # Recognition
            recogFlag = True
        elif np.linalg.norm(pp-p1, ord=2)<100:      # Clear
            canvas = np.full((768,1280,3), [255,255,255], np.uint8)
        else:
            mouseX = x
            mouseY = y
    if event == cv2.EVENT_LBUTTONUP:
        if mouseX!=-1 and mouseY!=-1:
            cv2.line(canvas, (mouseX, mouseY), (x, y), (0,0,0), thinkness)
        mouseX = -1
        mouseY = -1
    if event == cv2.EVENT_RBUTTONDOWN:
        canvas = np.full((768,1280,3), [255,255,255], np.uint8)
    if event == cv2.EVENT_MOUSEMOVE:
        if mouseX!=-1 and mouseY!=-1:
            cv2.line(canvas, (mouseX, mouseY), (x, y), (0,0,0), thinkness)
            mouseX = x
            mouseY = y
    cv2.imshow('canvas', canvas)
    cv2.waitKey(1)

def onTrackbar(x):
    global threshold
    threshold = x
    print(x)


def drawUI(image):
    cv2.circle(image, (0               , 0), 100, (   0, 255, 255), -1)
    cv2.circle(image, (image.shape[1]-1, 0), 100, (   0, 255,   0), -1)
    cv2.putText(image, 'RECOGNIZE', (10                ,20), cv2.FONT_HERSHEY_PLAIN, 1, (  0,   0,   0), 1)
    cv2.putText(image, 'CLEAR'    , (image.shape[1]-60 ,20), cv2.FONT_HERSHEY_PLAIN, 1, (  0,   0,   0), 1)

canvas = np.full((768,1280,3), [255,255,255], np.uint8)
threshold = 0.7

def main():
    global canvas
    global threshold
    global recogFlag

    # Plugin initialization
    ie = IECore()

    # text-detection-0003  in: 1,3,768,1280  
    # out:
    #  model/link_logits_/add   (1, 16, 192, 320)
    #  model/segm_logits/add,   (1,  2, 192, 320)
    model='text-detection-0003'
    model = './intel/'+model+'/FP16/'+model
    net_td = ie.read_network(model+'.xml', model+'.bin')
    input_blob_td = next(iter(net_td.inputs))
    out_blob_td = next(iter(net_td.outputs))
    exec_net_td = ie.load_network(net_td, 'CPU')

    # handwritten-japanese-recognition
    model = 'handwritten-japanese-recognition-0001'
    model = './intel/'+model+'/FP16/'+model
    net = ie.read_network(model+'.xml', model+'.bin')
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    characters = get_characters('data/kondate_nakayosi_char_list.txt')
    codec = CTCCodec(characters)
    input_batch_size, input_channel, input_height, input_width= net.inputs[input_blob].shape
    exec_net = ie.load_network(net, 'CPU')

    cH, cW, cC = canvas.shape

    cv2.namedWindow('canvas', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('canvas', onMouse)
    cv2.createTrackbar('Threshold', 'canvas', 70, 100, onTrackbar)

    cv2.imshow('canvas', canvas)
    cv2.waitKey(1)

    while True:
        key=0
        while key != ord(' ') and recogFlag==False:
            canvas2 = canvas.copy()
            drawUI(canvas2)
            key=cv2.waitKey(100)
            if key==27:
                return
            cv2.imshow('canvas', canvas2)

        recogFlag = False
        print('text detection')
        img = cv2.resize(canvas, (1280, 768))
        img = img.transpose((2,0,1))
        img = img.reshape((1,3,768,1280))
        res_td = exec_net_td.infer(inputs={input_blob_td: img})
        link = res_td['model/link_logits_/add']     # 1,16,192,320
        segm = res_td['model/segm_logits/add' ]     # 1, 2,192,320
        rects = text_detection_postprocess(link, segm, (1280, 768), threshold/100., threshold/100.)
        print('text detection - completed')

        canvas2 = canvas.copy()
        for rect in rects:
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.polylines(canvas2, [box], True, (0,255,255), 4)
            cv2.imshow('canvas', canvas2)

            most_left_idx, most_left = topLeftPoint(box)
            crop = cropRotatedImage(canvas, box, most_left_idx)

            # Read and pre-process input image (NOTE: one image only)
            #input_image = preprocess_input(canvas, input_height, input_width)[None,:,:,:]
            input_image = preprocess_input(crop, input_height, input_width)[None,:,:,:]

            # Start sync inference
            preds = exec_net.infer(inputs={input_blob: input_image})
            preds = preds[out_blob]
            result = codec.decode(preds)
            print('OCR result: ', result)
        print("done")
    return

if __name__ == '__main__':
    main()
