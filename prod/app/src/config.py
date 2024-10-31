import os

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIDEO_DATA_FOLDER = os.path.join(BASE_DIR, "data","video_data")

MODEL_CHECKPOINT = os.path.join(BASE_DIR, "model")
CUDA_DEVICE = "cuda"
PLAY_THRESHOLD=0.5

REDIS_CONT_NAME = "redis2_container"
MYSQL_CONT_NAME = "mysql2_container"
MYSQL_DB_NAME = "mydb"
MYSQL_DB_USER = "root"
MYSQL_DB_PWD = "password"

CELERY_BROKER_URL = f'redis://{REDIS_CONT_NAME}:6379/0'  # Redis 브로커 설정
CELERY_RESULT_BACKEND = f'redis://{REDIS_CONT_NAME}:6379/0'  # 작업 상태 저장을 위한 백엔드
CELERY_TIME_LIMIT = 12*60*60 # 12hr
CELERY_WORKER = 8
CELERY_MAX_MEMORY_PER_CHILD = 500000 # 단위는 KB (예: 500000 = 500MB)
CELERY_MAX_RETRY = 3
CELERY_RETRY_DELAY = 5 # 초 단위


HIGHTLIGHTS_NUM = 5
POST_CONS_CHUNK_SIZE = 3

TASK_STAT_CREATED = 'created'
TASK_STAT_COMPLETED = 'completed'
TASK_STAT_DOWNLOADED = 'downloaded'
TASK_STAT_CANCELED = 'cancelled'
TASK_STAT_ERROR = 'error'
