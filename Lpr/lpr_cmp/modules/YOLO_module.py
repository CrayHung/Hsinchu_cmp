from ctypes import *
import numpy as np
import cv2
import os
import time



if os.name == "nt":
    ''' load dll '''
    cwd = os.path.dirname(__file__)
    os.environ['PATH']=cwd+ ';'+os.environ['PATH']
    windll = os.path.join(cwd,"yolo_cpp_dll.dll")
    lib=CDLL(windll, RTLD_GLOBAL) # 3.7: RTLD_GLOBAL, 3.8: winmode=0
else:
    ''' load .so file '''
    lib = CDLL(os.path.join(os.getcwd(),"libdarknet.so"), RTLD_GLOBAL)
    # print('-- load libdarknet.so done --')  # 做成 .so, 在 import 時就會執行


''' interface between c and python '''


####################################################################
#
class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

#
class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("best_class_idx", c_int),  #??
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),       # YOLOv4需要,v3不需要
                ("points", c_int),              # YOLOv4需要,v3不需要
                ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]


#
class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

#
class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

#
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

#
copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]


#
make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

#
get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)


#
free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]


#
load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p


#
do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]


#
load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA



#
predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)



class YOLODetector():

    def __init__(self,
                 file_cfg_yolo,
                 file_weights_yolo,
                 file_data_yolo,
                 thresh_yolo,
                 nms=.45,
                 whitelist=[]):

        _to_byte = lambda name: ( name  if (name is bytes) else  name.encode('utf8') )
        self._net = load_net(_to_byte(file_cfg_yolo), _to_byte(file_weights_yolo), 0)
        self._meta = load_meta(_to_byte(file_data_yolo))
        self.whitelist = whitelist
        self.thresh_yolo = thresh_yolo
        self.nms = nms
        self._height_yolo = lib.network_height(self._net)
        self._width_yolo  = lib.network_width( self._net)
        self._image_darknet = make_image(self._width_yolo,self._height_yolo,3)
        image = np.zeros((self._height_yolo, self._width_yolo, 3), dtype=np.uint8)
        self.detect(image)

    def detect(self, image):
        height_image, width_image = image.shape[:2]
        image_bytes = self._preprocess(image)
        results = self._get_boxes(image_bytes, thresh=self.thresh_yolo, nms=self.nms)
        return self._convert_json(results, height_image, width_image)

    def _preprocess(self, image):
        image_resized = cv2.resize(image, (self._width_yolo,self._height_yolo))
        image_bytes = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB).tobytes()
        return image_bytes

    def _get_boxes(self,
                   image_bytes,
                   thresh=.5,
                   hier_thresh=.5,
                   nms=.45):

        num = c_int(0)
        pnum = pointer(num)
        copy_image_from_bytes(self._image_darknet, image_bytes)
        predict_image(self._net, self._image_darknet)
        letter_box = 0
        dets = get_network_boxes(self._net,
                                 self._image_darknet.w,
                                 self._image_darknet.h,
                                 thresh, hier_thresh,
                                 None, 0, pnum, letter_box)
        num = pnum[0]

        if nms:
            do_nms_sort(dets, num, self._meta.classes, nms)     # self._meta.classes

        results = []

        for j in range(num):
            for i in range(self._meta.classes):
                if dets[j].prob[i] > 0:
                    box = dets[j].bbox
                    results.append(( self._meta.names[i].decode('utf8'),
                                    dets[j].prob[i],
                                    (box.x, box.y, box.w, box.h) ))
        # print(results)
        results = sorted(results, key=lambda x: x[1], reverse=True)
        free_detections(dets, num)
        return results



    def _convert_coord(self, coord, height_image, width_image):
        x_center, y_center, x_width, y_height = coord
        x_center = float(x_center) * width_image  / self._width_yolo
        y_center = float(y_center) * height_image / self._height_yolo
        x_width  = float(x_width ) * width_image  / self._width_yolo
        y_height = float(y_height) * height_image / self._height_yolo
        left     = np.clip( x_center - x_width /2. , 0 , width_image  )
        top      = np.clip( y_center - y_height/2. , 0 , height_image )
        right    = np.clip( x_center + x_width /2. , 0 , width_image  )
        bottom   = np.clip( y_center + y_height/2. , 0 , height_image )
        return int(round(left)),\
               int(round(top)),\
               int(round(right-left)),\
               int(round(bottom-top))

    def _convert_json(self, results, height_image, width_image):
        # collect detection results with the same bounding box together
        dict_results = {}
        for result in results:
            name = result[0]
            if name in self.whitelist or len(self.whitelist)==0:
                key = result[2]
                dict_results.setdefault(key,[]).append((result[0],result[1]))
    
        bboxes = []
        for key in dict_results.keys():
            bbox = {}
            # key is coordinate of bounding box
            objectPicX, objectPicY,\
            objectWidth, objectHeight = self._convert_coord(key, height_image, width_image)
            if objectWidth==0 or objectHeight==0:
                break
            bbox["objectPicX"  ] = objectPicX
            bbox["objectPicY"  ] = objectPicY
            bbox["objectWidth" ] = objectWidth
            bbox["objectHeight"] = objectHeight
            objectTypes, confidences = zip(*dict_results[key])
            bbox["objectTypes"] = list(objectTypes)
            bbox["confidences"] = [round(c*100.,3) for c in confidences]
            bboxes.append(bbox)
        return bboxes



def init_yolo():
    thresh_yolo = 0.5
    nms = 0.45
    # file_cfg_yolo = "yolov4/y4.cfg" 
    # file_weights_yolo = "yolov4/y4.weights"
    # file_data_yolo = "yolov4/y4.data"
    file_cfg_yolo = "cfg/yolov4-tiny.cfg" 
    file_weights_yolo = "weights/yolov4.weights"
    file_data_yolo = "cfg/plate.data"
    whitelist = []


    yolo = YOLODetector(file_cfg_yolo=file_cfg_yolo,
                        file_weights_yolo=file_weights_yolo,
                        file_data_yolo=file_data_yolo,
                        thresh_yolo=thresh_yolo,
                        nms=nms,
                        whitelist=whitelist)

    return yolo

