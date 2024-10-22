from celery import Celery

# Celery 설정
celery_app = Celery(
    "video_tasks",
    broker="redis://redis_container:6379/0",  # Redis 브로커 설정
    backend="redis://redis_container:6379/0",  # 작업 결과 백엔드
)

celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
)

# tasks 모듈 임포트 (작업 등록)
import app.tasks  # 이 부분을 추가하여 작업이 등록되도록 함
