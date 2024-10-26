#!/bin/bash

# 4. Celery worker, Uvicorn 서버, gradio_app.py 실행 (백그라운드 실행, 로그 파일 추가)
echo "Starting Celery worker, Uvicorn server, and gradio_app.py..."

> /base/logs/celery/celery.log
> /base/logs/gradio/gradio.log
> /base/logs/uvicorn/uvicorn.log

/opt/conda/envs/myenv/bin/celery -A celery_app worker --loglevel=warning --logfile=/base/logs/celery/celery.log &

/opt/conda/envs/myenv/bin/python3 gradio_app.py >> /base/logs/gradio/gradio.log 2>&1 &
/opt/conda/envs/myenv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 9001

# /opt/conda/envs/myenv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 9001 >> /base/logs/uvicorn/uvicorn.log 2>&1 &
# /opt/conda/envs/myenv/bin/python3 gradio_app.py

echo "Celery worker, Uvicorn server, and gradio_app.py started in the background."

# 5. 종료 메시지 출력
echo "All processes started successfully."
