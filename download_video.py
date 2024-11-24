import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 웹캠, 영상 파일의 경우 이것을 사용하세요.:


cap = cv2.VideoCapture(0)
output_dir_path = 'output'
os.makedirs(output_dir_path,exist_ok=True) 

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    recording = False 
    out = None 
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라를 찾을 수 없습니다.")
            # 동영상을 불러올 경우는 'continue' 대신 'break'를 사용합니다.
            continue

        # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # 포즈 주석을 이미지 위에 그립니다.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        if recording  : 
            out.write(image)
        
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # 's' 키로 녹화 시작/중지
            if not recording:
                # 비디오 저장을 위한 VideoWriter 객체 생성
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(os.path.join(output_dir_path,'output.mp4'), fourcc, 20.0, (image.shape[1], image.shape[0]))
                recording = True
                print("녹화 시작")
            else:
                recording = False
                out.release()  # 비디오 파일 저장
                out = None
                print("녹화 중지 및 저장됨")
        elif key == ord('q') : 
            break 
    if recording  : 
        out.release()
        recording = False 
    cap.release()
    cv2.destroyAllWindows()
       