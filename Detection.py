import numpy as np
import cv2
import dlib

#얼굴 인식 학습모델 불러오기
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def detect(gray, frame):
    # 얼굴을 인식해 좌표로 랜드마크를 찾지만 눈의 위치만 출력해서 알려주는 함수

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    # xml파일을 이용해 얼굴을 찾는다.

    for (x, y, w, h) in faces:

        # OpenCV의 이미지를 dlib용 사각형으로 변환
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        # 모자이크 크기를 결정하기 위한 변수 선언
        v = 40
        #모자이크해주는 코드 5줄
        roi_gray = gray[y: y + h, x: x + w]
        roi_color = frame[y: y + h, x: x + w]
        roi = cv2.resize(roi_color, (w // v, h // v))
        roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)
        frame[y:y + h, x:x + w] = roi

    return frame


# 웹캠에서 실시간 이미지 가져오기
video_capture = cv2.VideoCapture(0)
# 파일을 쓰고 싶을때
#video_capture = cv2.VideoCapture('sample.mp4')

while True:
    # 웹캠 이미지를 프레임으로 자름, 그리고 무한루프로 계속 보여줌으로서 동영상화
    _, frame = video_capture.read()

    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴을 찾아 출력해주는 함수 호출
    canvas = detect(gray, frame)

    # 찾은 이미지 보여주기
    cv2.imshow("DETECTION_CGH",canvas)

    # q를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 끝
video_capture.release()
cv2.destroyAllWindows()

