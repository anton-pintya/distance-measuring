import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np

cam = cv2.VideoCapture(0)

wCam, hCam = 640, 480
cam.set(3, wCam)
cam.set(4, hCam)

detector = FaceMeshDetector(maxFaces=1)

f = 575
real_length = 16


def main():
    while True:
        _, img = cam.read()
        img = cv2.resize(img, (wCam, hCam))
        img = cv2.flip(img, 1)
        imgBlack = np.zeros_like(img)

        img, faces = detector.findFaceMesh(img, draw=False)

        if faces:
            face = faces[0]
            p10 = face[10]
            p152 = face[152]
            length, info = detector.findDistance(p10, p152)

            distance = int(f * real_length / length)

            if distance < 43:
                red = 25 * (distance - 21)
                cvzone.putTextRect(img, f'Distance: {distance} cm', (p10[0] - 100, p10[1] - 10), 1, 2,
                                   colorR=(0, 0, red))
            else:
                green = 15 * (-distance + 77)/2
                cvzone.putTextRect(img, f'Distance: {distance} cm', (p10[0] - 100, p10[1] - 10), 1, 2,
                                   colorR=(0, green, 0))

            scale = int((distance - 25) / 5)
            thickness = int((distance - 25) / 5)
            cvzone.putTextRect(imgBlack, 'Hello, World!', (50, 50), scale, thickness)
            try:
                imgStack = cvzone.stackImages([img, imgBlack], 2, 1)
            except:
                imgStack = img

            cv2.imshow("Image", imgStack)
        
        if cv2.waitKey(1) == 27:
            break


if __name__ == "__main__":
    main()
