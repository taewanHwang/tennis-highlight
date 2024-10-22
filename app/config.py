import os

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_FOLDER = os.path.join(BASE_DIR, "temp")
PROCESSED_VIDEOS_FOLDER = os.path.join(BASE_DIR, "processed_videos")
PLOT_FOLDER = os.path.join(BASE_DIR, "plot")
AUTH_FILEPATH = os.path.join(BASE_DIR, "users.txt")

MODEL_CHECKPOINT = "/base/model"
CUDA_DEVICE = "cuda"
PLAY_THRESHOLD=0.5

CELERY_BROKER_URL = 'redis://redis_container:6379/0'  # Redis 브로커 설정
CELERY_RESULT_BACKEND = 'redis://redis_container:6379/0'  # 작업 상태 저장을 위한 백엔드

# cp -r /home/disk5/taewan/study/practice2/demo/videomae-base-finetuned-kinetics-finetuned-playing-notplaying-training92/checkpoint-1200/* /home/disk5/taewan/study/practice2/demo3/model/
