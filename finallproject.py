from djitellopy import tello
import cv2
import cvzone
#detector1
from cvzone.HandTrackingModule import HandDetector
#detector2
from cvzone.PoseModule import PoseDetector

detector1 = PoseDetector()
detector2 = HandDetector(maxHands=2)

drone = tello.Tello()


xPID = cvzone.PID([0.22, 0, 0.1], 640 // 2)
yPID = cvzone.PID([0.27, 0, 0.1], 480 // 2, axis=1)
zPID = cvzone.PID([0.005, 0, 0.003], 12000, limit=[-20, 15])  

myPlotX = cvzone.LivePlot(yLimit=[-100, 100], char='X')
myPlotY = cvzone.LivePlot(yLimit=[-100, 100], char='Y')
myPlotZ = cvzone.LivePlot(yLimit=[-100, 100], char='Z')

drone.connect()
drone.streamon()
drone.takeoff()
drone.move_up(70)

while True:
    img = drone.get_frame_read().frame
    img = cv2.resize(img, (640, 480))
    bboxs, img = detector2.findHands(img)
    
    print(bboxs)
    
    xVal = 0
    yVal = 0
    zVal = 0
    if bboxs:
        cx, cy = bboxs[0]['center']
        x, y, w, h = bboxs[0]['bbox']
        area = w * h

        xVal = int(xPID.update(cx))
        yVal = int(yPID.update(cy))
        zVal = int(zPID.update(area))
        
        imgPlotX = myPlotX.update(xVal)
        imgPlotY = myPlotY.update(yVal)
        imgPlotZ = myPlotZ.update(zVal)

        img = xPID.draw(img, [cx, cy])
        img = yPID.draw(img, [cx, cy])
        imgStacked = cvzone.stackImages([img], 1, 0.75)
        
    else:
        imgStacked = cvzone.stackImages([img], 1, 0.75)   
    drone.send_rc_control(0, -zVal, -yVal, xVal)
    img = detector1.findPose(imgStacked, draw=True)  
 
    cv2.imshow("Image", img) 
    if cv2.waitKey(5) & 0xFF == ord('q'): 
        break