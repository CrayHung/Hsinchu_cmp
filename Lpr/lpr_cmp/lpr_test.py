from modules import YOLO_module
from modules import Align_onnx_module, OCR_onnx_module, labels_module
import cv2
import time
from datetime import datetime
from scipy.spatial import distance

def setWindow(w, h):
    cv2.namedWindow('demo', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('demo', w, h)

def show(image):
    cv2.imshow('demo', image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        exit()


def draw_labels(image0, labels):
    height_image, width_image = image0.shape[:2]
    number_row = 10
    width_show = int(width_image/(number_row+1))
    height_show = width_show//2

    image_labeled = image0.copy()
    for i, label in enumerate(labels):
        #objectTypes  = label.get("objectTypes")
        plateNumber = label.get("plateNumber", "")
        confidence = label.get("plateConfidence")
        objectPicX = label.get("objectPicX")
        objectPicY = label.get("objectPicY")
        objectWidth = label.get("objectWidth")
        objectHeight = label.get("objectHeight")
        array8 = label.get("perspectiveMatrix")
        matrix = ocr._array8_to_matrix3x3(array8)
        # text = (plateNumber + ': ' + str(round(confidence, 1))) if plateNumber else ''
        text = plateNumber if plateNumber else ''
        coord_opencv = (objectPicX,
                        objectPicY,
                        objectPicX + objectWidth,
                        objectPicY + objectHeight)

        label_generator.draw_bbox_with_text(
            image_labeled,
            text=text,
            size=3,
            _class=0,
            bbox_coord=coord_opencv
        )

        subimage_unaligned = image0[objectPicY:objectPicY+objectHeight,
                                    objectPicX:objectPicX+objectWidth, :]

        subimage_unaligned = cv2.resize(subimage_unaligned, FROM_SHAPE)
        subimage_aligned = cv2.warpPerspective(subimage_unaligned, matrix, TO_SHAPE)

        index_x = i % number_row
        index_y = i//number_row
        x0 = index_x*width_show
        y0 = index_y*width_show
        x1 = x0 + width_show
        y1 = y0 + height_show
        image_labeled[y0:y1, x0:x1, :] = cv2.resize(subimage_aligned, (width_show, height_show))

    return image_labeled



# ** load module
# yolo model config.
thresh_yolo = 0.45
nms = 0.4 # non-max-suppress
file_cfg_yolo = "modules/cfg/yolov4-tiny.cfg"
file_weights_yolo = "modules/weights/yolov4.weights"
file_data_yolo = "modules/cfg/plate.data"
whitelist = []
# alignment model config.
SIZE_IMAGE = 96
FROM_SHAPE = (SIZE_IMAGE, SIZE_IMAGE)
TO_SHAPE = (128,         64)
NUMBER_CORNER = 4
file_weights_lpa = "modules/weights/party.onnx"
# ocr model config.
padding = 0.08
thresh_ocr = 0.45
height_subimage = 64
width_subimage = 128
file_weights_ocr = "modules/weights/swag.onnx"

# load module
ocr = OCR_onnx_module.Recognition(file_weights_ocr=file_weights_ocr,
                                    padding=padding,
                                    thresh_ocr=thresh_ocr,
                                    height_subimage=height_subimage,
                                    width_subimage=width_subimage,
                                    aligned=True,
                                    from_shape=FROM_SHAPE)
yolo = YOLO_module.YOLODetector(file_cfg_yolo=file_cfg_yolo,
                                file_weights_yolo=file_weights_yolo,
                                file_data_yolo=file_data_yolo,
                                thresh_yolo=thresh_yolo,
                                nms=nms,
                                whitelist=whitelist)
align = Align_onnx_module.Alignment(file_weights_lpa=file_weights_lpa,
                                    from_shape=FROM_SHAPE,
                                    to_shape=TO_SHAPE,
                                    number_corner=NUMBER_CORNER)                                 


label_generator = labels_module.LabelGenerator(1)


url = "rtsp://admin:123456@192.168.103.48:554/onvif-media/media.amp?streamprofile=Profile1&audio=0"
vidcap = cv2.VideoCapture(url)
# 或此處處理重連

img_width = int(vidcap.get(3))
img_height = int(vidcap.get(4))
print(f'width = {img_width}, height = {img_height}')

# 
dis_width = int(img_width/2)
dis_height = int(img_height/2)

cnt = 0
item = {
    "recoged_c": {},    # current recog result
    "recoged_p": {}     # previous recog result
}

'''
variables
'''
valid_bnd = []
result_filter = []      # result fiter
valid_plate = []        # for valid plate number
recoged_diff = []       
recoged_p_key = []
time_record = ""
time_report = ""
time_file = ""
bnd_padding = 10        # boundary padding
tmp_list = []           # tmp list for image write


record_video = False    # record video?
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# fps = 10
# rec_width = 1920
# rec_height = 1080
# t_now = datetime.today()
# t_file = time_record.strftime("%Y%m%d_%H%M%S%f")
# avi_file = f'./video/{t_file}.avi'
# frmRec = cv2.VideoWriter(avi_file, fourcc, fps, (rec_width, rec_height))
# rec_cnt = 0


while True:

    time_record = datetime.today()
    time_report = time_record.strftime("%Y-%m-%d %H:%M:%S")
    time_file = time_record.strftime("%Y%m%d_%H%M%S%f")

    # get frame
    ret, img = vidcap.read()

    if ret: # frame is ready

        ''' record video '''
        # if record_video:
        #     frmRec.write(img)
        #     rec_cnt += 1
        #     if rec_cnt >= 3000: # 5 min, fps=10, 5*60*10 = 3000
        #         break

        # init state
        valid_bnd.clear()
        result_filter.clear()
        valid_plate.clear()
        item["recoged_c"].clear()
        recoged_diff.clear()
        recoged_p_key.clear()

        # yolo
        result_yolo = yolo.detect(img)  # result_yolo is a list []

        if len(result_yolo) > 0:

            ''' valid bbox '''
            valid_bnd = [res for res in result_yolo if (res["objectPicX"] > bnd_padding
                                                        and res["objectPicY"] > bnd_padding
                                                        and (res["objectPicX"] + res["objectWidth"]) < (img_width - bnd_padding)
                                                        and (res["objectPicY"] + res["objectHeight"]) < (img_height - bnd_padding))]
            
            ''' there is no valid bbox '''
            if len(valid_bnd) == 0:
                continue            
            

            ''' align and ocr '''
            result_align = align.predict(img, valid_bnd)
            result_ocr = ocr.predict(img, result_align)

            # []

            ''' any plateNumber is detected? '''
            result_filter = [item for item in result_ocr if 'plateNumber' in item]


            ''' there is no plateNumber '''
            if len(result_filter) == 0:
                continue


            ''' any valid plate number? '''
            # valid_plate = [res for res in result_filter if (res["plateNumber"].count("-") == 1
            #                                                 and res["plateNumber"].find("-") != 0
            #                                                 and res["plateNumber"].rfind("-") != (len(res["plateNumber"]) -1)
            #                                                 )]
            
            ''' there is no valid plate number '''
            # if len(valid_plate) == 0:
            #     continue


            ''' some valid plate number are detected'''
            for p in result_filter:
                plate_num = p["plateNumber"]

                centx = int((p["objectPicX"] + p["objectWidth"]/2))
                centy = int((p["objectPicY"] + p["objectHeight"]/2))

                file_path = f'./jpg/{time_file}_{plate_num}.jpg'


                ''' record plate '''
                item["recoged_c"].update({      # 當前frame
                    plate_num: plate_num
                })
                

                if plate_num in item["recoged_p"]:
                    ''' if plate already exists '''
                    item["recoged_p"][plate_num]["cnt"] = 0
                    pre_x = item["recoged_p"][plate_num]["cent"][0]
                    pre_y = item["recoged_p"][plate_num]["cent"][1]
                    dist_cur = distance.euclidean((centx, centy), (dis_width, dis_height))
                    dist_pre = distance.euclidean((pre_x, pre_y), (dis_width, dis_height))
                    if dist_pre <= dist_cur:
                        pass
                    else:
                        item["recoged_p"][plate_num]["cent"][0] = centx
                        item["recoged_p"][plate_num]["cent"][1] = centy
                        # save image
                        tmp_list.append(p)
                        img_p = draw_labels(img, tmp_list)
                        cv2.imwrite(item["recoged_p"][plate_num]["path"], img_p, [cv2.IMWRITE_JPEG_QUALITY, 100])
                        tmp_list.clear()
                else:
                    ''' a new plate number '''
                    item["recoged_p"].update({
                        plate_num: {
                            "path": file_path,
                            "cent": [centx, centy],
                            "cnt": 0
                        }
                    })
                    # write image
                    tmp_list.append(p)
                    img_p = draw_labels(img, tmp_list)
                    cv2.imwrite(file_path, img_p, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    print(f'new plate: {plate_num}')
                    tmp_list.clear()

            # img = draw_labels(img, result_ocr)


        ''' update recoged_p '''
        recoged_diff = list(item["recoged_p"].keys() - item["recoged_c"].keys())
        for key in recoged_diff:
            item["recoged_p"][key]["cnt"] += 1
        
        recoged_p_key = list(item["recoged_p"].keys())

        for p_key in recoged_p_key:
            if item["recoged_p"][p_key]["cnt"] > 300:
                item["recoged_p"].pop(p_key)

    else:
        # break
        ''' try to reconnect '''
        while True:
            videoCap.release()
            time.sleep(2)
            videoCap = cv2.VideoCapture(f'{url}')
            if videoCap.isOpened():
                print(f'reconnected')
                break
            else:
                print(f'try reconnect again')

