import math
import numpy as np
import cv2
from .pupil import Pupil

class Eye(object):
    """
    This class creates a new frame to isolate the eye and initiates the pupil detection.
    dlib를 통해 인식한 얼굴에서 눈을 추출하고, 동공을 탐지하는 클래스
    """

    #68개 좌표 중
    LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41] 
    RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

    def __init__(self, original_frame, landmarks, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None

        self._analyze(original_frame, landmarks, side, calibration)

    @staticmethod
    def _middle_point(p1, p2): #p refers to pixels
        """Returns the middle point (x,y) between two points
        Arguments: p1 (dlib.point): First point, p2 (dlib.point): Second point
        x, y는 p1과 p2의 중점인데, 눈의 윗부분과 아랫부분의 중점을 구하기 위해서임.
        """
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)
        return (x, y)

    def _isolate(self, frame, landmarks, points):
        """Isolate an eye, to have a frame without other part of the face.
        Arguments:
            frame (numpy.ndarray): Frame containing the face landmarks 
            (dlib.full_object_detection): Facial landmarks for the face region points 
            (list): Points of an eye (from the 68 Multi-PIE landmarks)
        인식한 얼굴에서 eye 부분만 떼기 위함.
        eye에 landmark을 두고 landmark의 x, y를 array에 넣어 region이라고 지정함.
        #양쪽 눈의 정확한 부분들의 좌표들을 array에 넣음.
        """
        region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        region = region.astype(np.int32)
        self.landmark_points = region
    
        # 얼굴에서 눈 부분만 추출하기 위해 mask 사용
        """mask는 원래의 frame 크기와 동일한 것으로 사용"""
        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8) 
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 255, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        """눈의 위치만 잘라내기
        min, max의 x, y를 array에서 가져옴"""
        margin = 5
        min_x = np.min(region[:, 0]) - margin 
        max_x = np.max(region[:, 0]) + margin 
        min_y = np.min(region[:, 1]) - margin 
        max_y = np.max(region[:, 1]) + margin 

        """x와 y의 min부터 max까지만 frame을 줘서 눈 부분만 frame을 덮음"""
        self.frame = eye[min_y:max_y, min_x:max_x] 
        self.origin = (min_x, min_y) 

        """eye만 나오는 frame을 보고싶다면 실행
        cv2.imshow("Eye", self.frame)"""

        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)

    def _blinking_ratio(self, landmarks, points):
        """Calculates a ratio that can indicate whether an eye is closed or not.
         It's the division of the width of the eye, by its height.
        Arguments:
            landmarks (dlib.full_object_detection): Facial landmarks for the face region points 
            (list): Points of an eye (from the 68 Multi-PIE landmarks)
        Returns: The computed ratio
        """
        """
        인식한 눈에 총 6개의 landmarks를 찍고, 가장 왼쪽을 0, 시계방향으로 돌아서 5까지.
        가장 왼쪽과 가장 오른쪽을 연결하는 수평선을 찍어서 이를 width라고 함.
        눈의 윗부분의 landmarks의 중점과 눈의 아랫부분의 landmarks을 middle_point라고 함.
        middle_point를 연결시키는 수직선을 찍어서 이를 height라고 함.
        눈을 감았을 때와 눈을 떴을 떄는 수평선과 수직선의 비율을 통해 알 수 있음.
        감았을 때 : 수평선 / 수직선 > 7
        떴을 때 : 수평선 / 수직선 < 7
        """

        left = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
        right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
        top = self._middle_point(landmarks.part(points[1]), landmarks.part(points[2]))
        bottom = self._middle_point(landmarks.part(points[5]), landmarks.part(points[4]))
        
        """hypot은 hypotenuse의 줄임말로 빗변을 구할 때 쓰는 함수
        left[0], right[0] = x, left[1], right[1] = y
        0 = x, 1 = y
        """
        eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))
        
        #print(eye_width / eye_height)

        try:
            ratio = eye_width / eye_height
        except ZeroDivisionError:
            ratio = None
        return ratio

    def _analyze(self, original_frame, landmarks, side, calibration):
        """Detects and isolates the eye in a new frame, sends data to the calibration and initializes Pupil object.
        Arguments:
            original_frame (numpy.ndarray): Frame passed by the user landmarks (dlib.full_object_detection): Facial landmarks for the face region
            side: Indicates whether it's the left eye (0) or the right eye (1)      
            calibration (calibration.Calibration): Manages the binarization threshold value
        """
        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return

        self.blinking = self._blinking_ratio(landmarks, points)
        self._isolate(original_frame, landmarks, points)

        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)
