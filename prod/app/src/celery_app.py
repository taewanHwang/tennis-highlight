from celery import Celery
import config

# Celery 설정
celery_app = Celery(
    "video_tasks",
    broker=f"redis://{config.REDIS_CONT_NAME}:6379/0",  # Redis 브로커 설정
    backend=f"redis://{config.REDIS_CONT_NAME}:6379/0",  # 작업 결과 백엔드
)

celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    worker_concurrency=config.CELERY_WORKER,  # 동시에 실행할 작업 수 제한
    task_soft_time_limit=int(config.CELERY_TIME_LIMIT*0.8),  # 소프트 리밋: 8초 이후 경고 발생
    task_time_limit=int(config.CELERY_TIME_LIMIT)       # 하드 리밋: 10초 이후 강제 종료
)

celery_app.conf.worker_max_memory_per_child = config.CELERY_MAX_MEMORY_PER_CHILD 
celery_app.conf.task_max_retries = config.CELERY_MAX_RETRY  # 각 작업은 최대 3번까지 재시도
celery_app.conf.task_retry_delay = config.CELERY_RETRY_DELAY  # 재시도 간의 지연 시간(초) 설정


# tasks 모듈 임포트 (작업 등록)
import tasks  # 이 부분을 추가하여 작업이 등록되도록 함
