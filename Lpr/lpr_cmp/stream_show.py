import cv2 
import time

def show_conf():
    cv2.namedWindow('demo', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('demo', 1280, 720)

def show(image):
    cv2.imshow('demo', image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        exit()



# -----------------------
#         cam url
# -----------------------
url = "rtsp://admin:123456@192.168.103.48:554/onvif-media/media.amp?streamprofile=Profile1&audio=0"
cap = cv2.VideoCapture(url)

# recon

cnt = 0
turns = True

show_conf()


while True:
    if cnt >=2000:
        break
    ret, frm = cap.read()
    if ret:
        cnt += 1
        
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
