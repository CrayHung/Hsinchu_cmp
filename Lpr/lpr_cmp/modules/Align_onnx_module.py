#!/usr/bin/python
import numpy as np
import cv2
import time

import string
from copy import deepcopy
import os
#import torch
import onnx
import caffe2.python.onnx.backend as backend

class Alignment():

    def __init__(self,
                 file_weights_lpa,
                 from_shape,
                 to_shape,
                 number_corner,
                 device = 'CPU',
                 batch_mode = False):

        assert ( device in ['CUDA', 'CPU'] ), f'Device {device} not supported'
        self.device = device

        self.model = self._onnx_loader(file_weights_lpa)
        self.from_shape = from_shape
        self.to_shape = to_shape
        self.number_corner = number_corner
        self.batch_mode = batch_mode
        self._corner_aligned = np.float32([[         0.,         0.],
                                           [to_shape[0],         0.],
                                           [to_shape[0],to_shape[1]],
                                           [          0,to_shape[1]]])
        data = np.zeros((1,3,
                         from_shape[1],
                         from_shape[0]),dtype=np.float32)
        self._align_batch(data)

    def _onnx_loader(self, filename):
        model_torch = onnx.load(filename)
        engine = backend.prepare(model_torch, device = self.device)
        def model_onnx(image):
            return engine.run(image)
        return model_onnx

    def _reshape_output_np(self, tensor, eps=1e-7):
        batchsize,_,h_input,w_input = tensor.shape
        tensor = tensor.reshape(batchsize,self.number_corner*4,-1)
        def _sigmoid(x):
            return 1./(np.exp(-x)+1.)
        def _softmax(x):
            x_exp = np.exp(x)
            return x_exp / np.sum(x_exp, axis=-1, keepdims=True)
        tensor_sigmoid = _sigmoid(tensor[:,:self.number_corner*2,:])
        tensor_softmax = _softmax(tensor[:,self.number_corner*2:,:])
    
        prediction = ( np.sum(tensor_sigmoid*tensor_softmax, axis=-1) * 3. - 1. ).reshape(batchsize,self.number_corner,2)
        mask = tensor_softmax.reshape(batchsize,self.number_corner*2,h_input,w_input)
        return prediction, np.sum(mask,axis=1)

    def _align_batch(self, data):
        outputs = self.model(data)[0]
        predictions, masks = self._reshape_output_np(outputs)
        matrices_perspective = self._corner_to_matrix_batch(predictions*np.float32(self.from_shape)[None,None,:])
        return matrices_perspective

    def _align_iterative(self, data):
        number_plate = len(data)
        outputs = []
        for n in range(number_plate):
            output = self.model( np.take(data,[n],0) )[0]
            outputs.append(output)
        outputs = np.concatenate(outputs, axis=0)
        predictions, masks = self._reshape_output_np(outputs)
        matrices_perspective = self._corner_to_matrix_batch(predictions*np.float32(self.from_shape)[None,None,:])
        return matrices_perspective

    def _corner_to_matrix_batch(self, corners):
        matrices = []
        for corner_unaligned in corners:
            matrix = cv2.getPerspectiveTransform(src=corner_unaligned,
                                                 dst=self._corner_aligned)
            array8 = self._matrix3x3_to_array8(matrix)
            matrices.append(array8)
        return matrices

    @staticmethod
    def _matrix3x3_to_array8(matrix3x3):
        array8 = matrix3x3.reshape(-1)[:-1]
        return list(array8)

    @staticmethod
    def _array8_to_matrix3x3(array8):
        matrix3x3 = np.ones(9, dtype=np.float32)
        matrix3x3[:-1] = array8
        return matrix3x3.reshape(3,3)

    def predict(self, image, result_yolo):
        height_image, width_image = image.shape[:2]
        number_plate = len(result_yolo)
        if number_plate == 0:
            return result_yolo
        subimages = np.zeros((number_plate,
                              self.from_shape[1],
                              self.from_shape[0],3),dtype=np.uint8)
        for i,bbox in enumerate(result_yolo):
            left = bbox["objectPicX"]
            top  = bbox["objectPicY"]
            h_bbox = bbox["objectHeight"]
            w_bbox = bbox["objectWidth"]
            bottom = min( top+h_bbox, height_image)
            right  = min(left+w_bbox, width_image )
            top    = max( top, 0)
            left   = max(left, 0)
            subimage = image[top:bottom, left:right, :]
            subimages[i,:,:,:] = cv2.resize( subimage , self.from_shape )
        data = np.transpose(subimages,(0,3,1,2))
        if self.batch_mode:
            matrices_perspective = self._align_batch(data)
        else:
            matrices_perspective = self._align_iterative(data)
        for i in range(number_plate):
            result_yolo[i]["perspectiveMatrix"] = matrices_perspective[i]
        return result_yolo

    def show_alignment(self, image, result_alignment):
        height_image, width_image = image.shape[:2]
        for i,bbox in enumerate(result_alignment):
            left = bbox["objectPicX"]
            top  = bbox["objectPicY"]
            h_bbox = bbox["objectHeight"]
            w_bbox = bbox["objectWidth"]
            matrix = bbox['perspectiveMatrix']
            matrix = self._array8_to_matrix3x3(matrix)
            bottom = min( top+h_bbox, height_image)
            right  = min(left+w_bbox,  width_image)
            top    = max( top, 0)
            left   = max(left, 0)
            subimage = image[top:bottom, left:right, :]
            subimage_unaligned = cv2.resize( subimage, self.from_shape )
            subimage_aligned = cv2.warpPerspective(subimage_unaligned, matrix, self.to_shape)
            show(subimage_aligned)

def show(image):
    cv2.imshow('demo', image)
    key = cv2.waitKey(0)
    if key == ord('q'):
        exit()

# if __name__ == "__main__":
#     import YOLO_module

#     #os.environ["QT_X11_NO_MITSHM"] = "1"

#     ''' yolo model '''
#     # settings
#     thresh_yolo = 0.1
#     nms = 0.1

#     # yolov3-tiny for license plate
#     file_cfg_yolo = "cfg/yolov3_tiny_plate.cfg"
#     file_weights_yolo = "weights/yolov3_tiny_plate_final.weights"
#     file_data_yolo = "cfg/plate.data"

#     # prediction with some constraint
#     # if whitelist is empty, detect everything
#     #whitelist = ["car","motorbike","bus", "truck"]
#     whitelist = []

#     ''' alignment model '''
#     # settings
#     SIZE_IMAGE = 96
#     FROM_SHAPE = (SIZE_IMAGE, SIZE_IMAGE)
#     TO_SHAPE   = (       128,         64)
#     NUMBER_CORNER = 4

#     # ocr weights
#     file_weights_lpa = "weights/party.onnx"

#     # prepare models
#     yolo = YOLO_module.YOLODetector(file_cfg_yolo     = file_cfg_yolo,
#                                     file_weights_yolo = file_weights_yolo,
#                                     file_data_yolo    = file_data_yolo,
#                                     thresh_yolo       = thresh_yolo,
#                                     nms               = nms,
#                                     whitelist         = whitelist)

#     align = Alignment(file_weights_lpa = file_weights_lpa,
#                       from_shape       = FROM_SHAPE,
#                       to_shape         = TO_SHAPE,
#                       number_corner    = NUMBER_CORNER)

#     # prediction
#     for _ in range(100):

#         # prediction starts
#         time0 = time.time()

#         image = cv2.imread("data/5978-YA.jpg")

#         # YOLO
#         result_yolo = yolo.detect(image)
#         print(result_yolo)

#         result_align = align.predict(image, result_yolo)
#         print(result_align)

#         #show(image)
#         #align.show_alignment(image, result_align)

#         # prediction ends
#         print("duration %.6f s"%(time.time()-time0))

#     del yolo
#     del align
