import time
import os
import queue
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, date
import copy
import websocket
import base64
import json
import aiohttp
import asyncio
from scipy.spatial import distance

sys.path.append('.')

def lpr_det(que):

    from modules import YOLO_module
    from modules import Align_onnx_module, OCR_onnx_module
    import logging
    # import modules.OCR_onnx_module


    def show_conf():
        cv2.namedWindow('lpr_demo', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('lpr_demo', 1280, 720)

    def show(image):
        cv2.imshow('lpr_demo', image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            exit()

    def array8_to_matrix3x3(array8):
        matrix3x3 = np.ones(9, dtype=np.float32)
        matrix3x3[:-1] = array8
        return matrix3x3.reshape(3,3)

    def draw_bbox(frm, labels, frm_num):
        frm_labeled = frm.copy()
        FROM_SHAPE = (96, 96)
        TO_SHAPE = (128, 64)
        cv2.putText(frm_labeled, f'frame: {(frm_num+1):03}', (int(ROIw/2), int(ROIh - 40)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 225, 235), 3, cv2.LINE_AA)

        for i, label in enumerate(labels):
            # plateNumber = label.get("plateNumber", "")
            objX = label.get("objectPicX")
            objY = label.get("objectPicY")
            objW = label.get("objectWidth")
            objH = label.get("objectHeight")
            array8 = label.get("perspectiveMatrix")
            matrix = array8_to_matrix3x3(array8)

            plate_img = frm_labeled[objY:objY+objH, objX:objX+objW]
            plate_img_unaligned = cv2.resize(plate_img, FROM_SHAPE)
            plate_img_aligned = cv2.warpPerspective(plate_img_unaligned, matrix, TO_SHAPE)

            plate_img_aligned = cv2.resize(plate_img_aligned, (256, 128), interpolation=cv2.INTER_AREA) 
            frm_labeled[0:128, (ROIw-256):ROIw] = plate_img_aligned

            # left = max(objX-objW*0.02, 0)
            # top = max(objY-objH*0.02, 0)
            # right = min(objX + objW*1.04, ROIw)
            # bottom = min(objY + objH*1.04, ROIh)

            cv2.rectangle(frm_labeled, (objX, objY), (objX + objW, objY + objH), (0, 0, 250), 2)
            # cv2.rectangle(frm_labeled, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 250), 2)
        return frm_labeled


    async def aiohttp_post(rep_url, resp_para):
        headers = {'content-type': 'application/json'}
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(rep_url, json=resp_para, headers=headers) as resp:
                    pass
            except Exception as e:
                err_log = f'Unknown exception: {e} {datetime.today().strftime("%Y%m%d_%Hh%Mm%Ss")}'
                print(err_log)


    # ** load module
    # yolo model config.
    thresh_yolo = 0.3
    nms = 0.1
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
    thresh_ocr = 0.3
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

    # # ws
    # websocket.enableTrace(False)
    # ws = websocket.WebSocket()
    # try:
    #     ws.connect("ws://61.216.140.11:9098/ws", header={"ID": "MOT"})
    # except:
    #     print("ws connection fail")

    
    ROIx = 0
    ROIy = 0
    ROIw = 912  #912
    ROIh = 912
    ROIw_ws = 480
    ROIh_ws = 480
    imgw = 1920
    imgh = 1080
    thumb_w = 128
    thumb_h = 128
    plate_vote = {}
    vote_key = []           # 供暫存, 找出現次數最高的車牌號
    vote_val = []           # 供暫存, 找出現次數最高的車牌號
    plate_most = ""
    idx = 0
    img_roi = np.zeros((ROIh, ROIw, 3))
    img_roi_ws = np.zeros((ROIh_ws, ROIw_ws, 3))
    img_thumbnail = np.zeros((thumb_w, thumb_h, 3))


    img_none = None
    img_roi_none = None

    image_fname = ""
    image_path = ""
    image_roi_path = ""
    image_thumb_path = ""

    # ** set display window
    # show_conf()

    while True:
        try:
            # get data from queue
            data = que.get(block=True, timeout=None)
            # print(f'que get data = {data}')

            # connect
            vidcap = cv2.VideoCapture(data["video_path"])

            while True:
                ret, img = vidcap.read()
                if ret:
                    # print(f'idx = {idx}')
                    area = {}
                    if len(data["area"][idx]) > 0:
                        area = data["area"][idx]
                        # print(area["x0"], area["y0"], area["x1"], area["y1"])
                        
                        # **extract area:
                        centx = int((area["x0"] + area["x1"])/2)
                        centy = int((area["y0"] + area["y1"])/2)
                        # print(area["x0"], area["y0"], area["x1"], area["y1"])
                        if (centx - ROIw/2) < 0:
                            ROIx = 0
                        elif (centx + ROIw/2) > imgw:
                            ROIx = int(imgw - ROIw)
                        else:
                            ROIx = int(centx - ROIw/2)

                        if (centy - ROIh/2) < 0:
                            ROIy = 0
                        elif (centy + ROIh/2) > imgh:
                            ROIy = int(imgh - ROIh)
                        else:
                            ROIy = int(centy - ROIh/2)

                        img_roi = img[ROIy:ROIy+ROIh, ROIx:ROIx+ROIw]  # **912*912 ROI 影像
                        # print(centx, centy, ROIy, ROIy+ROIh, ROIx, ROIx+ROIw)

                        # print(img_roi.shape)
                       
                        result_yolo = yolo.detect(img_roi)
                        for index in range(len(result_yolo)):
                            result_yolo[index]['objectPicX'] = max(result_yolo[index]['objectPicX'] - int(result_yolo[index]['objectWidth']*0.02), 0)
                            result_yolo[index]['objectPicY'] = max(result_yolo[index]['objectPicY'] - int(result_yolo[index]['objectHeight']*0.02), 0)
                            result_yolo[index]['objectWidth'] = min(int(result_yolo[index]['objectWidth']*1.04), ROIw-result_yolo[index]['objectPicX'])
                            result_yolo[index]['objectHeight'] = min(int(result_yolo[index]['objectHeight']*1.04), ROIh-result_yolo[index]['objectPicY'])
                        # print(result_yolo)  # [{'objectPicX': 364, 'objectPicY': 465, 'objectWidth': 92, 'objectHeight': 63, 'objectTypes': ['Plate'], 'confidences': [94.862]}]
                        
                        result_align = align.predict(img_roi, result_yolo)
                        result_ocr = ocr.predict(img_roi, result_align)

                        result_filter = [item for item in result_ocr if 'plateNumber' in item]
                        if len(result_filter) == 0: # **no plate number is detected
                            if img_none is None:
                                img_none = img
                                img_roi_none = img_roi
                        else:
                            # **detect some plate number
                            # **過濾出有效車號
                            result_valid = [res for res in result_filter if (res["plateNumber"].count("-") == 1
                                                                        and res["plateNumber"].find("-") != 0
                                                                        and res["plateNumber"].rfind("-") != (len(res["plateNumber"]) - 1)
                                                                        )]
                            
                            # **draw
                            img_roi = draw_bbox(img_roi, result_ocr, idx)
                            # show(img_roi)

                            # 保留距離中心點最近的一個
                            if len(result_valid) > 0:
                                dist = []
                                result_final = []
                                if len(result_valid) == 1:
                                    result_final.append(result_valid[0])
                                else:
                                    # 計算各車牌到中心點距離
                                    for res_i in result_valid:
                                        obj_x = res_i['objectPicX'] + res_i['objectWidth']/2
                                        obj_y = res_i['objectPicY'] + res_i['objectHeight']/2
                                        d = distance.euclidean((int(obj_x), int(obj_y)), (int(ROIw/2), int(ROIh/2)))  # (x, y)
                                        dist.append(d)
                                    min_idx = dist.index(min(dist))
                                    result_final.append(result_valid[min_idx])

                                # **plate_vote
                                p = result_final[0]["plateNumber"]
                                if p not in plate_vote:
                                    plate_vote.update({
                                        p: {
                                            "cnt": 1,
                                            "img": img,
                                            "img_roi": img_roi
                                        }
                                    })
                                else:
                                    plate_vote[p]["cnt"] += 1

                    else:
                        # **可不用
                        img_roi = img[ROIy:ROIy+ROIh, ROIx:ROIx+ROIw]  # **912*912 ROI 影像
                        img_roi = draw_bbox(img_roi, [], idx)
                        # show(img)

                    idx += 1
                    
                    # ** websocket, 前端顯示
                    # img_roi_ws = cv2.resize(img_roi, (ROIw_ws, ROIh_ws), interpolation=cv2.INTER_AREA)
                    # _, img_buf = cv2.imencode('.jpg', img_roi_ws)
                    # img_b64 = base64.b64encode(img_buf)
                    # img_byte = img_b64.decode()
                    # payload = json.dumps({
                    #     "id": data["cam_det_id"],
                    #     "img": img_byte,
                    #     "plate": ""
                    # })
                    
                    # try:
                    #     ws.send(payload)
                    # except Exception as e:
                    #     print(f'ws.send error: {e}')

                    # area["x0"], area["y0"], area["x1"], area["y1"], area["cx"], area["cy"]

                else:   # **recog is done
                    # plate number vote result
                    if len(plate_vote) > 0:
                        for k, v in plate_vote.items():
                            # for example, k = "ABC-1234", v = {'cnt': 3, 'img': [...]}
                            vote_key.append(k)
                            vote_val.append(v["cnt"])

                        max_idx = np.argmax(vote_val)
                        plate_most = vote_key[max_idx]

                        image_fname = f'{data["video_name"].split(".")[0]}_{plate_most}'
                        image_path = f'./video/{image_fname}.jpg'
                        image_roi_path = f'./video/{image_fname}_roi.jpg'
                        image_thumb_path = f'./video/{image_fname}_thumbnail.jpg'
                        img_thumbnail = cv2.resize(plate_vote[plate_most]["img_roi"], (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(image_path, plate_vote[plate_most]["img"], [cv2.IMWRITE_JPEG_QUALITY, 100])
                        cv2.imwrite(image_roi_path, plate_vote[plate_most]["img_roi"], [cv2.IMWRITE_JPEG_QUALITY, 100])
                        cv2.imwrite(image_thumb_path, img_thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    else:
                        plate_most = "NULL"
                        image_fname = f'{data["video_name"].split(".")[0]}_{plate_most}'
                        image_path = f'./video/{image_fname}.jpg'
                        image_roi_path = f'./video/{image_fname}_roi.jpg'
                        image_thumb_path = f'./video/{image_fname}_thumbnail.jpg'
                        img_thumbnail_none = cv2.resize(img_roi_none, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(image_path, img_none, [cv2.IMWRITE_JPEG_QUALITY, 100])
                        cv2.imwrite(image_roi_path, img_roi_none, [cv2.IMWRITE_JPEG_QUALITY, 100])
                        cv2.imwrite(image_thumb_path, img_thumbnail_none, [cv2.IMWRITE_JPEG_QUALITY, 100])

                    
                    # ** websocket, 前端顯示\
                    # img_roi_ws = cv2.resize(img_roi, (ROIw_ws, ROIh_ws), interpolation=cv2.INTER_AREA)    # 1280, 720
                    # _, img_buf = cv2.imencode('.jpg', img_roi_ws)
                    # img_b64 = base64.b64encode(img_buf)
                    # img_byte = img_b64.decode()
                    # payload = json.dumps({
                    #     "id": data["cam_det_id"],
                    #     "img": img_byte,
                    #     "plate": plate_most
                    # })
 
                    # try:
                    #     ws.send(payload)
                    # except Exception as e:
                    #     print(f'ws.send error: {e}')

                    # post
                    resp_data_195 = {
                        "RoadName": "Nanliao",
                        "EventType": "1",
                        "ReportDate": data["timestamp"],
                        "CarType": "motorcycle",
                        "PlateNumber": plate_most,
                        "ImgPath": f'{image_fname}.jpg',
                        'ImgRoiPath': f'{image_fname}_roi.jpg',
                        'ImgThumbPath': f'{image_fname}_thumbnail.jpg',
                        "VideoPath": data["video_name"]     # record_20230217_021341.mp4
                    }

                    resp_data_local = {
                        "cameraId": data["cam_det_id"],
                        "reportDate": data["timestamp"],
                        "plateNumber": plate_most,
                        "imgPath": f'{image_fname}.jpg',
                        'imgRoiPath': f'{image_fname}_roi.jpg',
                        'imgThumbPath': f'{image_fname}_thumbnail.jpg',
                        "videoPath": data["video_name"]     # record_20230217_021341.mp4
                    }

                    print(f'resp_data_local = {resp_data_local}')

                    tasks = [aiohttp_post(f'http://61.216.140.11:9098/event', resp_data_195),
                             aiohttp_post(f'http://localhost:8080/lpr/event', resp_data_local)]

                    loop = asyncio.get_event_loop()
                    # loop.run_until_complete(aiohttp_post(f'http://61.216.140.11:9098/event', resp_data))
                    # loop.run_until_complete(aiohttp_post(f'http://localhost:8080/lpr/event', resp_data))
                    loop.run_until_complete(asyncio.wait(tasks))
                    # localhost:8080/lpr/event

                    # **reset status
                    vidcap.release()
                    idx = 0
                    vote_key.clear()
                    vote_val.clear()
                    plate_vote.clear()
                    img_none = None
                    img_roi_none = None
                    break
        except queue.Empty:
            pass
        except Exception as e:
            print(e)




'''
print(f'que get data = {data}')
{
    'time': '20230209_215654', 
    'path': './video/0005.avi', 
    'area': [{'x0': 638, 'y0': 358, 'x1': 929, 'y1': 852, 'cx': 783, 'cy': 728}, 
             {'x0': 640, 'y0': 360, 'x1': 907, 'y1': 816, 'cx': 773, 'cy': 702}, 
             {'x0': 627, 'y0': 348, 'x1': 877, 'y1': 773, 'cx': 752, 'cy': 666}
             ...]
}
'''

'''
(res["plateNumber"].count("-") == 1 and res["plateNumber"].find("-") != 0 and res["plateNumber"].rfind("-") != (len(res["plateNumber"]) -1))
'''

'''
t1 = time.time()
t2 = time.time()
td = t2 - t1
print(f'{td*1e3:4.2f} ms')
'''

'''
resp_data = {
    'RoadName': 'Nanliao', 
    'EventType': '1', 
    'ReportDate': '2023-02-26 18:09:03', 
    'CarType': 'motorcycle', 
    'PlateNumber': '933-NGD', 
    'ImgPath': 'live1_0001_20230226_180903_933-NGD.jpg', 
    'ImgRoiPath': 'live1_0001_20230226_180903_933-NGD_roi.jpg', 
    'ImgThumbPath': 'live1_0001_20230226_180903_933-NGD_thumbnail.jpg', 
    'VideoPath': 'live1_0001_20230226_180903.mp4'
}
'''