import cv2 
import time

def show_conf():
    cv2.namedWindow('demo', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('demo', 640, 360)

def show(image):
    cv2.imshow('demo', image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        exit()


# 槍機: stream url
# url_1 = 'rtsp://admin:123456@106.1.93.8:10007/media/media.amp?streamprofile=Profile1&audio=0'
# url_2 = 'rtsp://admin:123456@106.1.93.8:10009/onvif-media/media.amp?streamprofile=Profile1&audio=0'


url_1 = "rtsp://admin:123456@106.1.93.8:10008/onvif-media/media.amp?streamprofile=Profile1&audio=0"
# url_2 = "rtsp://admin:123456@106.1.93.8:10006/onvif-media/media.amp?streamprofile=Profile1&audio=0"
# url_2 = "rtsp://admin:@192.168.103.116:554/stream/CH2"    # cam2
# url_2 = "rtsp://admin:@192.168.103.116:554/stream/CH3"  # cam1

url_2 = "rtsp://admin:@192.168.103.116:554/stream/ch3?replay=1?start=1678611185?end=1678611210"
avi_file = f'./video/rec_20230312_165309.avi'


fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 12
rec_width = 1920
rec_height = 1080

frmRec = cv2.VideoWriter(avi_file, fourcc, fps, (rec_width, rec_height))

# -----------------------
#         cam url
# -----------------------
url_cam = url_2

cap = cv2.VideoCapture(url_cam)

cnt = 0
turns = True

show_conf()


while True:
    if cnt >=220:
        break
    ret, frm = cap.read()
    if ret:
        cnt += 1
        # ** video record
        frmRec.write(frm)

        # **resize for test
        # img_gun = cv2.resize(frm_gun, (1280, 720), interpolation=cv2.INTER_AREA)
        # frm_ball = cv2.resize(frm_ball, (640, 360), interpolation=cv2.INTER_AREA)
        
        show(frm)
        print(cnt)

    else:
        break
        # while True:
        #     cap.release()
        #     time.sleep(3)
        #     cap = cv2.VideoCapture(url_cam)
        #     if cap.isOpened():
        #         print(f're-connected')
        #         break
        #     else:
        #         print(f'fail to re-connect, try again')
