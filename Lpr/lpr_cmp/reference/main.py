import cv2
from tools.colors import _COLORS
import numpy as np
import multiprocessing as mp
import queue
import sys
import time
from datetime import datetime, date
import base64
import json
import websocket
import argparse
# from utils.visualize import plot_tracking
import copy 
import os

from main_lpr import lpr_det

sys.path.insert(0, 'Detection')
sys.path.insert(0, 'Tracking')


def mot_det(que_m, que, cam_url, cam_id, cam_det_id, typ):

    from Detection.darknet.detect import Darknet
    from Tracking.bytetrack import BYTETracker  
    # from Tracking.sort.tracking import Sort      


    def show_conf():
        cv2.namedWindow('demo', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('demo', 1280, 720)

    def show(image):
        cv2.imshow('demo', image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            exit()

    # **param
    # data_track = {
    #             "tlwh": online_tlwhs,       # [array([208.51348663, 417.43228534, 182.05381035, 476.80796337])]
    #             "ids": online_ids,          # [10]
    #             "score" : online_scores     # [0.97511]
    #         }
    def ObjTracking(draw_track, img, data_tracking):
        curr_obj = []

        for i in range(len(data_tracking["ids"])):
            # print(type(data_tracking["ids"][i]))      # int
            track_id = int(data_tracking["ids"][i])
            box = data_tracking["tlwh"][i][:4]
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = x0 + int(box[2])
            y1 = y0 + int(box[3])

            # 記錄當前frame的所有追蹤id
            curr_obj.append(track_id)

            # 先判斷是否在mot內
            if track_id not in mot:
                t_ent = datetime.today()
                t_record = t_ent.strftime('%Y%m%d_%H%M%S')
                t_record_db = t_ent.strftime('%Y-%m-%d %H:%M:%S')
                videoRec, video_name, video_path = create_recorder(f'{typ}_{track_id:04}_{t_record}')
                
                mot.update({
                    track_id: {
                        "traj":[{
                            # 座標基於 1920x1080
                            "x0": x0,
                            "y0": y0,
                            "x1": x1,
                            "y1": y1,
                            "cx": int((x0 + x1)/2),
                            "cy": int(y1 - (y1 - y0)*0.25),
                        }],
                        "state":{
                            "miss": 0,      # 記錄已經幾個frame無此track_id, 從mot刪除
                            "type": "Motorcycle", # 車種,  car, truck
                            "done": False,   # "done" => 舉發過並完成錄影
                            "que_put": False,
                            "tracked": 1
                        },
                        "rec":{
                            "recorder": videoRec,
                            "time": t_record,
                            "timestamp": t_record_db,
                            "video_path": video_path,       # ./video/record_20230217_021341.mp4
                            "video_name": video_name,       # record_20230217_021341.mp4
                            "counts": 0
                        }
                    }
                })
                # cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
            else:
                # 已在mot裡
                mot[track_id]["traj"].append({
                    # 座標基於 1920x1080
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "cx": int((x0 + x1)/2),
                    "cy": int(y1 - (y1 - y0)*0.25),
                })
                mot[track_id]["state"]["tracked"] += 1
                # cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        
        # **if len(data_tracking["ids"]) == 0, 上段略過, 於此處對mot內每一個key補上 traj
        # **if len(data_tracking["ids"]) > 0, 在上段中, 新key update, 舊key增加traj, 不在curr_obj的, 亦於此處補上 traj
        rm_list = []
        for item in mot:
            # **存影片
            if not mot[item]["state"]["done"] and mot[item]["state"]["tracked"] >= 2:
                mot[item]["rec"]["recorder"].write(img)
                mot[item]["rec"]["counts"] += 1
                if mot[item]["rec"]["counts"] >= 60:
                    mot[item]["rec"]["recorder"].release()
                    mot[item]["state"]["done"] = True

            if item not in curr_obj:
                last_traj = mot[item]["traj"][-1]
                mot[item]["traj"].append(last_traj)
                mot[item]["state"]["miss"] += 1
                if mot[item]["state"]["miss"] == 70:
                    mot[item]["rec"]["recorder"].release()
                    rm_list.append(item)
            else:
                mot[item]["state"]["miss"] = 0

            # if len(mot[item]["traj"]) > 60:
            #     mot[track_id]["traj"].pop(0)

        # ** 刪除消失過久的物件
        for rm in rm_list:
            print(f'remove id = {rm}, tracked len = {mot[rm]["state"]["tracked"]}')
            if mot[rm]["state"]["tracked"] < 2:
                try:
                    del_file = mot[rm]["rec"]["video_path"]
                    os.remove(del_file)
                    print(f'delete {del_file}')
                except OSError as e:
                    print(f'Fail to delete {del_file}: {e}')
            mot.pop(rm)

        rm_list.clear()


        # **基於mot狀況繪製軌跡
        if draw_track:
            for obj in mot: # obj為track_id
                # mot中, 某個track_id的記錄長度
                length = len(mot[obj]["traj"])
                
                if length < 2:
                    continue

                # ** 畫線
                # 起始點
                cx_p = 0
                cy_p = 0
                for k in range(length):
                    if k > 25:
                        continue
                    cent = (mot[obj]["traj"][length - 1 - k]["cx"], mot[obj]["traj"][length - 1 - k]["cy"])
                    if cx_p == 0 and cy_p == 0:
                        pass
                    else:
                        cv2.line(img, (cent[0], cent[1]), (cx_p, cy_p), (125, 250 - 1*k, 0), int(8 - k*0.08))
                    
                    cv2.circle(img, cent, int(12-(k*0.15)), (125, 250 - 1*k, 0), -1)
                    cx_p = cent[0]
                    cy_p = cent[1]

                '''
                cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
                座標 => 是指文字的左下角座標點
                字型 => 可用 opencv 內建字型, cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX
                線條種類 => 可用 cv2.LINE_AA
                
                '''
                text = "ID " + str(obj)
                cv2.putText(img, text, (mot[obj]["traj"][-1]["x0"], mot[obj]["traj"][-1]["y0"] - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 6, cv2.LINE_AA)

        # show(img)
        # ** websocket, 前端顯示
        # cv2.imshow("demo", img)
        # frame_ws = frame.copy()

        # t1 = time.time()     
        img = cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA)    # 1280, 720; 864, 486; 960, 540; 800, 450
        _, img_buf = cv2.imencode('.jpg', img)
        img_b64 = base64.b64encode(img_buf)
        img_byte = img_b64.decode()
        payload = json.dumps({
            "id": cam_id,
            "img": img_byte
        })
        
        try:
            ws.send(payload)
        except Exception as e:
            print(f'ws.send error: {e}')
        
        # t2 = time.time()
        # td = t2-t1
        # print(f'{td*1e3:4.2f} ms')


    def make_parser():
        parser = argparse.ArgumentParser(description='Bytes Track demo')
        parser.add_argument("-f", "--fps", type=int, default=10, required=False, help="FPS of the video")
        parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
        parser.add_argument("--track_buffer", type=int, default=50, help="the frames for keep lost tracks, usually as same with FPS")
        parser.add_argument("--match_thresh", type=float, default=0.5, help="matching threshold for tracking")
        parser.add_argument('--min_box_area', type=float, default=30, help='filter out tiny boxes')
        parser.add_argument(
            "--aspect_ratio_thresh", type=float, default=1.6,
            help="threshold for filtering out boxes of which aspect ratio are above the given value."
        )

        return parser


    def create_recorder(filename):  # filename => record_20230217_021341
        # video_file = f'./video/{filename}.avi'
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_file_save = f'./video/{filename}.mp4'
        video_file_name = f'{filename}.mp4'
        fourcc = int(cv2.VideoWriter_fourcc(*'avc1'))
        fps = 12
        rec_width = 1920
        rec_height = 1080
        frmRec = cv2.VideoWriter(video_file_save, fourcc, fps, (rec_width, rec_height))

        return frmRec, video_file_name, video_file_save

    # ----------------------------------------- 
    #          mot_det() main function
    # ----------------------------------------- 
    # show_conf()
    mot = {}
    draw_trk = True
    frm_width = 1920
    frm_height = 1080
    tbox_w = 400
    tbox_h = 400

    # **tracker
    args = make_parser().parse_args(args=[])
    tracker_bytetrack = BYTETracker(args, 12)
    # tracker_sort = Sort()

    # **detector
    detector = Darknet()
    label = detector.names
    videoCap = cv2.VideoCapture(f'{cam_url}')

    time.sleep(1)

    # **ws
    websocket.enableTrace(False)
    ws = websocket.WebSocket()
    try:
        ws.connect("ws://61.216.140.11:9098/ws", header={"ID": "MOT"})
    except:
        print("ws connection fail")

    time.sleep(3)
    while True:
        # time.sleep(0.2)
        ret, frame = videoCap.read()
        
        if ret:
            # t1 = time.time()
            # **原本的copy
            # if typ != "record":
                # frame = frm_source.copy()
            # else:
                # **1440*810(160,170 ~ 1600,980) => 1920*1080
            # frame_roi = frm_source[170:980, 160:1600]
            # frame = cv2.resize(frame_roi, (1920, 1080), interpolation=cv2.INTER_AREA)
            
            # frame = frm_source.copy()

            # **put for bgskip
            img_mog = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            try:
                que_m.put({
                    "frm_mog": img_mog,
                })
            except Exception as e:
                print(f'mot put exception: {e}')


            box_det, cls, confs = detector.detect(frame)
            
            # box_detects = np.array(box_det).astype(int)
            # classes = np.array(cls)
            # confids = np.array(confs)

            box_detect = []
            if len(box_det) > 0:
                for i in range(len(box_det)):
                    # x0, y0, x1, y1 => box_det[i][0], box_det[i][1]), (box_det[i][2], box_det[i][3]

                    # cv2.rectangle(frame, (box_det[i][0], box_det[i][1]), (box_det[i][2], box_det[i][3]), (0, 0, 200), 3)
                    bbox_centx = int((box_det[i][0] + box_det[i][2])/2)
                    bbox_centy = int((box_det[i][1] + box_det[i][3])/2)
                    # bbox_top = max(bbox_centy - 200, 0)
                    # bbox_left = max(bbox_centx - 200, 0)
                    # bbox_right = min(bbox_centx + 200, 1920)
                    # bbox_bottom = min(bbox_centy + 200, 1080)

                    if (bbox_centx - tbox_w/2) < 0:
                        bbox_left = 0
                    elif (bbox_centx + tbox_w/2) > frm_width:
                        bbox_left = int(frm_width - tbox_w)
                    else:
                        bbox_left = int(bbox_centx - tbox_w/2)

                    if (bbox_centy - tbox_h/2) < 0:
                        bbox_top = 0
                    elif (bbox_centy + tbox_h/2) > frm_height:
                        bbox_top = int(frm_height - tbox_h)
                    else:
                        bbox_top = int(bbox_centy - tbox_h/2)
                    # box_detect.append([box_det[i][0], box_det[i][1], box_det[i][2], box_det[i][3], confs[i][0]])
                    box_detect.append([bbox_left, bbox_top, bbox_left+tbox_w, bbox_top+tbox_h, confs[i][0]])
                    # cv2.rectangle(frame, (bbox_left, bbox_top), (bbox_right, bbox_bottom), (0, 140, 100), 3)
            else:
                box_detect.append([1, 1, 1, 1, 0.1])
            
            box = np.array(box_detect)

            data_track = tracker_bytetrack.update(box, [1080, 1920], (1080, 1920))

            online_tlwhs = []
            online_ids = []
            online_scores = []
            # results = []
            for target in data_track:
                tlwh = target.tlwh      # (x1, y1, w, h)
                tid = target.track_id
                # vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                # if tlwh[2] * tlwh[3] > args.min_box_area or vertical:
                if tlwh[2] * tlwh[3] > args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(target.score)
                    # save results
                    # results.append(f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{target.score:.2f},-1,-1,-1\n")

            '''
            print(f'online_tlwhs = {online_tlwhs}')
            print(f'online_ids = {online_ids}')
            print(f'online_scores = {online_scores}')

            online_tlwhs = [array([208.51348663, 417.43228534, 182.05381035, 476.80796337])]
            online_ids = [10]       # tracking id
            online_scores = [0.97511]
            '''

            # online_im = plot_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id, fps=0.0)
            # frame_id += 1
            # cv2.imshow("online_im", online_im)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            track_result = {
                "tlwh": online_tlwhs,       # [array([208.51348663, 417.43228534, 182.05381035, 476.80796337])]
                "ids": online_ids,          # [10]
                "score" : online_scores     # [0.97511]
            }

            # **drawing and update mot{}
            ObjTracking(draw_trk, frame, track_result)

            # **check mot and put
            # mot_list = mot.keys()
            # print(mot_list)
            for item in mot:
                if mot[item]["state"]["done"] and not mot[item]["state"]["que_put"] and mot[item]["state"]["tracked"] >= 2:
                    pdata_time = mot[item]["rec"]["time"]
                    pdata_timestamp = mot[item]["rec"]["timestamp"]
                    pdata_video_path = mot[item]["rec"]["video_path"]
                    pdata_video_name = mot[item]["rec"]["video_name"]
                    pdata_area = copy.deepcopy(mot[item]["traj"])
                    pdata = {
                            "cam_det_id": cam_det_id,
                            "time": pdata_time,
                            "timestamp": pdata_timestamp,
                            "video_path": pdata_video_path,       # ./video/record_20230217_021341.mp4
                            "video_name": pdata_video_name,       # record_20230217_021341.mp4
                            "area": pdata_area                    # for ROI
                        }

                    try:
                        que.put(pdata)
                        mot[item]["state"]["que_put"] = True
                    except Exception as e:
                        print(e)
            
            # t2 = time.time()
            # td = t2 - t1
            # print(f'{td*1e3:4.2f}  ms' )
        else:
            # **try to reconnect
            while True:
                videoCap.release()
                time.sleep(2)
                videoCap = cv2.VideoCapture(f'{cam_url}')
                if videoCap.isOpened():
                    print(f'reconnected')
                    break
                else:
                    print(f'try reconnect again')




def mog(que_mot, cid):

    def create_recorder(filename):  # filename => record_20230217_021341
        # video_file = f'./video/{filename}.avi'
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_file_save = f'./video_mog/{filename}.mp4'
        fourcc = int(cv2.VideoWriter_fourcc(*'avc1'))
        fps = 12
        rec_width = 640
        rec_height = 360
        frmRec = cv2.VideoWriter(video_file_save, fourcc, fps, (rec_width, rec_height))

        return frmRec

    mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    t_now = datetime.today()
    t_rec = t_now.strftime('%Y%m%d_%H%M%S')
    fname = f'{cid}_mog_{t_rec}'
    videoRec = create_recorder(fname)

    frm_mog2_last = None
    frm_mog2_gray = None
    frm_mog2_diff = None
    frm_mog2_res = None
    frm_mog2_ready = False

    ct = 0

    while True:
        try:
            data = que_mot.get(block = True)

            if frm_mog2_ready:

                frm_mog2_gray = cv2.cvtColor(data["frm_mog"], cv2.COLOR_BGR2GRAY)

                frm_mog2_diff = cv2.absdiff(frm_mog2_last, frm_mog2_gray)
                _, frm_mog2_diff = cv2.threshold(frm_mog2_diff, 40, 255, cv2.THRESH_TOZERO)

                frm_mog2_res = mog2.apply(frm_mog2_diff)
                frm_mog2_res = cv2.morphologyEx(frm_mog2_res, cv2.MORPH_OPEN, kernel)  
                frm_mog2_res = cv2.morphologyEx(frm_mog2_res, cv2.MORPH_CLOSE, kernel) 
                frm_mog2_res = cv2.morphologyEx(frm_mog2_res, cv2.MORPH_CLOSE, kernel)
                # frm_mog2_res = cv2.morphologyEx(frm_mog2_res, cv2.MORPH_CLOSE, kernel)

                # **update frm_last
                frm_mog2_last = frm_mog2_gray
                cnts, hierarchy = cv2.findContours(frm_mog2_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(cnts) != 0:
                    videoRec.write(data["frm_mog"])
                    ct += 1
                else:
                    pass

                if ct >= 4320:
                    videoRec.release()
                    videoRec = None
                    t_now = datetime.today()
                    t_rec = t_now.strftime('%Y%m%d_%H%M%S')
                    fname = f'{cid}_mog_{t_rec}'
                    videoRec = create_recorder(fname)
                    ct = 0

            # **MOG2 init frame 
            else:
                frm_mog2_last = cv2.cvtColor(data["frm_mog"], cv2.COLOR_BGR2GRAY)
                frm_mog2_ready = True
        except queue.Empty:
            pass
        except Exception as e:
            print(f'mog get error: {e}')


if __name__ == "__main__":

    mp.set_start_method(method='spawn')

    que_rec = mp.Queue()
    q1 = mp.Queue(80)
    q2 = mp.Queue(80)

    processes = []

    # source
    # rec_20230207_002.avi
    # rec_20230207_002_skipBg.avi

    # src_1 = {
    #     "url": "./video_demo/rec_20230218_001.avi",
    #     "cid": "CAM_10",
    #     "cid_det": "CAM_11"
    # }

    src_1 = {
        "url": "rtsp://admin:@192.168.103.116:554/stream/CH3",
        # "url": "rtsp://admin:123456@106.1.93.8:10008/onvif-media/media.amp?streamprofile=Profile1&audio=0",
        "cid": "CAM_10",
        "cid_det": "CAM_11"
    }
    src_2 = {
        "url": "rtsp://admin:@192.168.103.116:554/stream/CH2",
        # "url": "rtsp://admin:123456@106.1.93.8:10006/onvif-media/media.amp?streamprofile=Profile1&audio=0",
        "cid": "CAM_20",
        "cid_det": "CAM_21"
    }

    processes.append(mp.Process(target=mot_det, args=(q1, que_rec, src_1["url"], src_1["cid"], src_1["cid_det"], "live1",)))
    processes.append(mp.Process(target=mot_det, args=(q2, que_rec, src_2["url"], src_2["cid"], src_2["cid_det"], "live2",)))
    processes.append(mp.Process(target=lpr_det, args=(que_rec, )))
    processes.append(mp.Process(target=mog, args=(q1, src_1["cid"],)))
    processes.append(mp.Process(target=mog, args=(q2, src_2["cid"],)))

    for process in processes:
        process.daemon = True
        process.start()

    for process in processes:
        process.join()



# url_1 = "rtsp://admin:123456@106.1.93.8:10008/onvif-media/media.amp?streamprofile=Profile1&audio=0"
# url_2 = "rtsp://admin:123456@106.1.93.8:10006/onvif-media/media.amp?streamprofile=Profile1&audio=0"


# **multiprocessing.Event()
# 會全域性定義一個 Flag
# => set()方法 : 將 Flag 設定為True
# => clear()方法 : 將 Flag 設定為False
# => is_set()方法 : 判断当前的Flag的值
# => wait() : Flag為False，event.wait() 阻塞; Flag為True，event.wait() 不阻塞


# t1 = time.time()
# t2 = time.time()
# print(f'{(t2-t1)*1e3:6.2f} ms')


# def euclidean_distance(detection, tracked_object):
#     return np.linalg.norm(detection.points - tracked_object.estimate)

'''
範例:
(1) box_detects = [[   4  258  247  469]
                    [ 881   68  949  134]
                    [ 810   82  860  125]
                    [1313  119 1390  197]
                    [ 841   52  893  103]]
(2) classes = [[0]
                [2]
                [0]
                [2]
                [2]]
(3) confids = [[0.99981]
                [0.96772]
                [0.91574]
                [0.75416]
                [0.61029]]
'''

'''
Live
Main stream:
rtsp://admin:@192.168.103.116:554/stream/CH2
rtsp://admin:@192.168.103.116:554/stream/CH3

Replay:
取得2號攝影機的2021/6/28/15:10:0~2021/6/28/15:15:00的回放影像
rtsp://admin:@192.168.103.116:554/stream/ch2?replay=1?start=1624864200?end=1624864500

'''