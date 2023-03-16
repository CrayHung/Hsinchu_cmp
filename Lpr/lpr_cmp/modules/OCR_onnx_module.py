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

class Recognition():

    def __init__(self,
                 file_weights_ocr,
                 padding,
                 thresh_ocr,
                 height_subimage,
                 width_subimage,
                 device = 'CPU',
                 aligned = False,
                 from_shape = None,
                 batch_mode = False):

        if aligned:
            assert ( type(from_shape) is tuple and len(from_shape) == 2 ),\
            f'If aligned is True, from_shape is original shape of subimage'
        self.aligned = aligned
        self.from_shape = from_shape

        assert ( device in ['CUDA', 'CPU'] ), f'Device {device} not supported'
        self.device = device

        self.model = self._onnx_loader(file_weights_ocr)
        self.padding = padding
        self.thresh_ocr = thresh_ocr
        self.height_subimage = height_subimage
        self.width_subimage = width_subimage
        self.batch_mode = batch_mode

        # create strings for plate number encoding and decoding
        self.CHARACTERS = string.digits + string.ascii_uppercase + "-"

        # setting for postprocess
        self.NUMBER_CLASS = len(self.CHARACTERS)
        self.NUMBER_CLASS_NO_SEP = self.NUMBER_CLASS-1
        self.NUMBER_CHARACTER = 8

        data = np.zeros((1,3,
                         self.height_subimage,
                         self.width_subimage),dtype=np.uint8)
        self._predict_ocr_batch(data)

    def _onnx_loader(self, filename):
        model_torch = onnx.load(filename)
        engine = backend.prepare(model_torch, device = self.device)
        def model_onnx(image):
            return engine.run(image)
        return model_onnx
    
    def _reshape_output_np(self, tensor, eps=1e-7):
        batchsize, _, height_input, width_input = tensor.shape
        sigmoid = lambda x: 1. / (1.+np.exp(-x))
        tensor_sigmoid = sigmoid(tensor)
    
        heatmap = tensor_sigmoid[:,:self.NUMBER_CLASS_NO_SEP,:,:]
        mask    = tensor_sigmoid[:,self.NUMBER_CLASS_NO_SEP:,:,:]
    
        heatmap_1d = heatmap.reshape(batchsize,self.NUMBER_CLASS_NO_SEP,-1)
        mask_1d = mask.reshape(batchsize,self.NUMBER_CHARACTER,-1)
        mask_1d_normalized = mask_1d/(np.sum(mask_1d,axis=-1)+eps)[:,:,None]
        prediction_no_sep = np.sum( mask_1d_normalized[:,:,None,:] \
                                      * heatmap_1d[:,None,:,:] , axis=-1 )
        prediction_sep = 1.-np.max(mask_1d,axis=-1)[:,:,None]
        prediction = np.concatenate((prediction_no_sep, prediction_sep), axis=-1)
        return prediction, mask, heatmap

    def _predict_ocr_batch(self, data):
        outputs = self.model(data)[0]
        predictions, masks, heatmaps = self._reshape_output_np(outputs)
        plate_numbers, confidences = self._parse_plate(predictions)
        return plate_numbers, confidences

    def _predict_ocr_iterative(self, data):
        number_plate = len(data)
        outputs = []
        for n in range(number_plate):
            output = self.model(np.take(data, [n], axis=0))[0]
            outputs.append(output)
        outputs = np.concatenate(outputs, axis=0)
        predictions, masks, heatmaps = self._reshape_output_np(outputs)
        plate_numbers, confidences = self._parse_plate(predictions)
        return plate_numbers, confidences

    def _parse_plate(self, predictions):
        number_plate = len(predictions)
        argmax = np.argmax(predictions, axis=-1)
        plate_numbers = []
        confidences = []
        for i in range(number_plate):
            plate_numbers.append( ''.join( self.CHARACTERS[a]  for a in argmax[i] ) )
            confidences.append([ predictions[i,n,argmax[i,n]]  for n in range(self.NUMBER_CHARACTER) ])
        return plate_numbers, confidences

    def _warp_perspective(self, subimage_unaligned, matrix):
        subimage_unaligned = cv2.resize(subimage_unaligned, self.from_shape)
        subimage_aligned = cv2.warpPerspective(subimage_unaligned, matrix, (self.width_subimage,self.height_subimage))
        return subimage_aligned

    def predict(self, image, result_yolo):
        height_image, width_image = image.shape[:2]
        number_plate = len(result_yolo)
        if number_plate == 0:
            return result_yolo
        subimages = np.zeros((number_plate,
                              self.height_subimage,
                              self.width_subimage,3),dtype=np.uint8)
        for i,bbox in enumerate(result_yolo):
            left = bbox["objectPicX"]
            top  = bbox["objectPicY"]
            h_bbox = bbox["objectHeight"]
            w_bbox = bbox["objectWidth"]
            padding_y = round(h_bbox*self.padding)
            padding_x = round(w_bbox*self.padding)
            bottom = min( top+h_bbox+padding_y, height_image)
            right  = min(left+w_bbox+padding_x,  width_image)
            top    = max( top-padding_y, 0)
            left   = max(left-padding_x, 0)
            subimage = image[top:bottom, left:right, :]
            array8 = bbox.get("perspectiveMatrix", None)
            if self.aligned and array8 is not None:
                subimage = cv2.resize(subimage, self.from_shape)
                matrix = self._array8_to_matrix3x3(array8)
                subimage = self._warp_perspective(subimage, matrix)
            else:
                subimage = cv2.resize(subimage, (self.width_subimage,self.height_subimage))
            subimages[i,:,:,:] = subimage
        data = np.transpose(subimages,(0,3,1,2))
        if self.batch_mode:
            plate_numbers, confidences = self._predict_ocr_batch(data)
        else:
            plate_numbers, confidences = self._predict_ocr_iterative(data)
        for i in range(number_plate):
            plate_number = plate_numbers[i]
            plate_confidence = np.min(confidences[i])
            if plate_confidence < self.thresh_ocr:
                continue
            result_yolo[i]["plateNumber"] = plate_number.rstrip("-")
            result_yolo[i]["plateConfidence"] = round(plate_confidence*1e2,3)
        return result_yolo

    @staticmethod
    def _array8_to_matrix3x3(array8):
        matrix3x3 = np.ones(9, dtype=np.float32)
        matrix3x3[:-1] = array8
        return matrix3x3.reshape(3,3)



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

#     ''' ocr model '''
#     # settings
#     padding = 0.1
#     thresh_ocr = 0.5
#     height_subimage = 64
#     width_subimage = 128

#     # ocr weights
#     file_weights_ocr = "weights/swag.onnx"

#     # prepare models
#     ocr = Recognition(file_weights_ocr = file_weights_ocr,
#                       padding          = padding,
#                       thresh_ocr       = thresh_ocr,
#                       height_subimage  = height_subimage,
#                       width_subimage   = width_subimage)

#     yolo = YOLO_module.YOLODetector(file_cfg_yolo     = file_cfg_yolo,
#                                     file_weights_yolo = file_weights_yolo,
#                                     file_data_yolo    = file_data_yolo,
#                                     thresh_yolo       = thresh_yolo,
#                                     nms               = nms,
#                                     whitelist         = whitelist)

#     # prediction
#     for _ in range(100):

#         # prediction starts
#         time0 = time.time()

#         image = cv2.imread("data/5978-YA.jpg")

#         # YOLO
#         result_yolo = yolo.detect(image)
#         print(result_yolo)

#         result_ocr = ocr.predict(image, result_yolo)
#         print(result_ocr)

#         # prediction ends
#         print("duration %.6f s"%(time.time()-time0))

#     del yolo
#     del ocr
