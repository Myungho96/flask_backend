from flask import Flask, render_template, Response, request
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import torch
import os
import datetime
from tqdm.notebook import tqdm
import time
from facenet_pytorch import MTCNN
import shutil
import albumentations as A

app = Flask(__name__)

camera = cv2.VideoCapture(0)  # use 0 for web camera

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


@app.route('/Webcam')
def Webcam():
    return render_template('Webcam.html')


#업로드 HTML 렌더링
@app.route('/upload')
def render_file():
   return render_template('index.html')

#data set 만드는 AI

#오늘 날짜 가져오는 함수,--------- 폴더명을 날짜로 저장하기 위함!
def get_today():
    now = time.localtime()
    s= "%02d-%02d-%02d-%02d-%02d" %(now.tm_mon, now.tm_mday, now.tm_hour,now.tm_min,now.tm_sec)
    return s

#폴더를 만드는 함수
def createFolder(folder_name):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

#파일 업로드 처리
@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      #저장할 경로 + 파일명
      f.save("./DataSet/" + secure_filename(f.filename))
      mtcnn = MTCNN(keep_all=True, device = 'cuda:0' if torch.cuda.is_available() else 'cpu')
########################################################################
      # Load a video
      #얼굴 추출을 할 비디오를 가져오기
      #load the video
      #C:/Users/dearj/Pictures/Camera Roll/juwoovod6.mp4
      v_cap = cv2.VideoCapture('./DataSet/'+ f.filename)

      #get the grame count
      v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

      #폴더를 생성할 위치
      root_dir = "./DataSet/"

      #새로운 폴더 생성
      today = get_today()
      work_dir = root_dir + "/" + today
      createFolder(work_dir)

      frames = []
      count = 0

      print(v_len)

      for _ in range(v_len):
          count += 1
          #load the frame
          success, frame = v_cap.read()
          if not success:
              continue

          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          #10개 frame마다 하나씩 dataset으로 뽑아온다.
          if count % 10 == 0:
              frames.append(frame)


      #save_path = [f'image_{i}.jpg' for i in range((len(frames)))]
      save_path = [f'image_{i}.jpg' for i in range(50)]

      for frame, path in zip(frames, save_path):
          mtcnn(frame, save_path=path)


      #처음 이미지가 저장되는 위치로 설정
      source_dir = "."
      #이미지를 옮길 위치 설정
      dest_dir = "./DataSet" + "/" + today

      #source_dir 디렉토리에 있는 파일들을 검색하여 리스트에 집어넣음
      source_list = os.listdir(source_dir)

      #source_list 리스트에서 image_숫자.jpg로 끝나는 파일들을 dest_dir로 옮기기
      # 좀 더 효율적인 알고리즘이 있으면 변경바람(ex)이분탐색?
      for f in source_list:
          for i in range(len(frames)):
              tmp = "image_" + str(i) +".jpg"
              if(f == tmp):
                  shutil.move(f, dest_dir)
                  break

      return "Data set 생성"

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)