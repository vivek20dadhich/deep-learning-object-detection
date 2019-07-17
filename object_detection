import numpy as np
import cv2

net=cv2.dnn.readNetFromCaffe(r'C:\Users\ivive\Downloads\MobileNetSSD_deploy.prototxt',r'C:\Users\ivive\Downloads\MobileNetSSD_deploy.caffemodel')

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

v=cv2.VideoCapture(0)
while(True):
    r,frame=v.read()
    print(np.shape(frame))#(480,640,3)
    ri=cv2.resize(frame,(300, 300))

    blob=cv2.dnn.blobFromImage(ri,0.007843,(300, 300), 127.5,False)
    net.setInput(blob)
    detections = net.forward()
    print(detections.shape)

    (h, w) = frame.shape[:2]
    
    for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
     
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.3:
                    # extract the index of the class label from the `detections`,
                    # then compute the (x, y)-coordinates of the bounding box for
                    # the object
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
     
                    # display the prediction
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    print("[INFO] {}".format(label))
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                            COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    cv2.imshow("Image",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        cv2.destroyAllWindows()
        break
    print(detections)

