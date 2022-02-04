import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

cam = cv2.VideoCapture(0)

wCam, hCam = 640, 480
cam.set(3, wCam)
cam.set(4, hCam)

f = 575
real_length = 16

plotX = LivePlot(640, 480, [-wCam // 2, wCam // 2], invert=True)
plotY = LivePlot(640, 480, [-hCam // 2, hCam // 2], invert=True)
plotZ = LivePlot(640, 480, [0, 100], invert=True)

detector = FaceMeshDetector(maxFaces=1)


def main():
    while True:
        _, img = cam.read()
        img = cv2.resize(img, (wCam, hCam))
        img = cv2.flip(img, 1)

        img, faces = detector.findFaceMesh(img, draw=False)

        cv2.line(img, (wCam // 2, 0), (wCam // 2, hCam), (255, 0, 255), 2)
        cv2.line(img, (0, hCam // 2), (wCam, hCam // 2), (255, 0, 255), 2)

        cx, cy = wCam // 2, hCam // 2

        if faces:
            face = faces[0]
            nose = face[1]
            cv2.circle(img, nose, 1, (0, 255, 0), 2)

            distance, info = detector.findDistance(nose, (cx, cy))
            cv2.line(img, nose, (cx, cy), (255, 0, 255), 2)

            cvzone.putTextRect(img, f'Uncentered: {int(distance)}', (20, 20), 1, 2)

            up_right = nose[0] > cx and nose[1] < cy
            up_left = nose[0] < cx and nose[1] < cy
            down_right = nose[0] > cx and nose[1] > cy
            down_left = nose[0] < cx and nose[1] > cy

            conditions = {'Go Up Right': up_right,
                          'Go Up Left': up_left,
                          'Go Down Right': down_right,
                          'Go Down Left': down_left}

            for text, condition in conditions.items():
                if condition is True:
                    cvzone.putTextRect(img, text, (20, 100), 1, 2)

            p10 = face[10]
            p152 = face[152]
            length, info = detector.findDistance(p10, p152)

            distance = f * real_length / length

            if distance < 43:
                red = int(25 * (distance - 21))
                cvzone.putTextRect(img, f'Distance: {int(distance)} cm', (20, 60), 1, 2,
                                   colorR=(0, 0, red))
            else:
                green = int(15 * (-distance + 77) / 2)
                cvzone.putTextRect(img, f'Distance: {int(distance)} cm', (20, 60), 1, 2,
                                   colorR=(0, green, 0))

            imgX = plotX.update(nose[0] - wCam // 2)
            cvzone.putTextRect(imgX, 'X', (5, 20), 1, 2)

            imgY = plotY.update(-nose[1] + hCam // 2)
            cvzone.putTextRect(imgY, 'Y', (5, 20), 1, 2)

            imgZ = plotZ.update(distance)
            cvzone.putTextRect(imgZ, 'Z', (5, 20), 1, 2)

            imgStack1 = cvzone.stackImages([img, imgZ], 2, 0.7)
            imgStack2 = cvzone.stackImages([imgX, imgY], 2, 0.7)
            imgStack = cvzone.stackImages([imgStack1, imgStack2], 1, 1)

            cv2.imshow("Image", imgStack)

        if cv2.waitKey(1) == 27:
            break


if __name__ == "__main__":
    main()
