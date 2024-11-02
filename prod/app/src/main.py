from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List
from fastapi.responses import FileResponse
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from celery_app import celery_app
import logging
from models import YouTubeVideoInput

from datetime import datetime, timezone

from db import engine
import db_crud, db_models
import config
from utils import validate_youtube_video_input, validate_upload_video_input

# db_models.Base.metadata.drop_all(bind=engine)  # 기존 테이블 삭제
db_models.Base.metadata.create_all(bind=engine)  # 테이블 다시 생성

app = FastAPI()

# 로그 파일 설정
log_filename = "/base/logs/uvicorn/uvicorn.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s] %(message)s",
)

################################## Video processing function ##################################

@app.post("/process_youtube_video/")
async def process_youtube_video(input_data: YouTubeVideoInput):
    # 1. 입력 검증
    try:
        validate_youtube_video_input(input_data)
    except ValueError as e:
        error_message = str(e)
        print(f"Video processing input error: {error_message}")
        return {"message": f"Video processing input error, {error_message}"}
    
    # 2. Celery 작업으로 파일 처리 요청
    task = celery_app.send_task('app.src.tasks.process_youtube_video', args=[input_data.model_dump()])
    
    return {"message": "Video processing started", "task_id": task.id}
    
@app.post("/process_upload_video/")
async def process_upload_video(file: UploadFile = File(...), process_options: List[str] = Form(...), username: str = Form(...)):
    # 1. 입력 검증
    try:
        validate_upload_video_input(file, process_options, username)
    except ValueError as e:
        error_message = str(e)
        print(f"Video processing input error: {error_message}")
        return {"message": f"Video processing input error, {error_message}"}

    # 2. 파일을 서버에 저장
    video_path = os.path.join(config.VIDEO_DATA_FOLDER, username, 'temp',f"{file.filename}")
    print(f"upload video video_path:{video_path}")
    with open(video_path, "wb") as f:
        f.write(await file.read())
    
    # 3. Celery 작업으로 파일 처리 요청
    task = celery_app.send_task('app.src.tasks.process_uploaded_video', args=[video_path, process_options, username])
    
    return {"message": "Video file uploaded and processing started", "task_id": task.id}

################################## Task management function ##################################

@app.get("/check_status/{req_id}")
async def check_status(req_id: str):    
    # 함수화한 세션 사용
    task = db_crud.get_task_by_req_id(req_id)
    

    if not task:
        return {"error": "Task not found"}
    
    # 기본 응답 형식
    response = {
        "task_id": task.id,
        "status": task.status
    }

    # elapsed_time 계산 함수
    def calculate_elapsed_time(task):
        now = datetime.now(timezone.utc)
        elapsed_time = int((now - task.created_at.replace(tzinfo=timezone.utc)).total_seconds())
        if elapsed_time < 60:
            return f"{elapsed_time} sec"
        else:
            return f"{elapsed_time // 60} min"

    # 상태에 따른 메시지 설정
    if task.status in [config.TASK_STAT_DOWNLOADED, config.TASK_STAT_CREATED]:
        response["message"] = f"The task is in process (total time {calculate_elapsed_time(task)})"
    elif task.status == config.TASK_STAT_COMPLETED:
        response["message"] = f"The task is in completed (total time {calculate_elapsed_time(task)})"
    elif task.status == config.TASK_STAT_CANCELED:
        response["message"] = "The task is canceled."
    elif task.status == config.TASK_STAT_ERROR:
        response["message"] = "The task has error."
    else:
        response["message"] = "Unknown task state."

    return response


@app.get("/download_video/{task_id}")
async def download_video(task_id: int, video_type: str = "full"):
    """
    task_id로 비디오 처리 상태를 확인한 후, 성공한 경우 결과 비디오 파일을 제공하는 엔드포인트.

    Parameters:
    - task_id: 작업의 task_id
    - video_type: "full", "playing", "highlights", "segments" 등
    """
    print(f"task_id: {task_id}, video_type: {video_type}", flush=True)

    # DB에서 작업 조회
    video_task = db_crud.get_video_task_by_id(task_id)
    print(f"video_task: {video_task}", flush=True)
    print(f"video_task.status: {video_task.status}", flush=True)

    # 상태 확인
    if video_task.status != config.TASK_STAT_COMPLETED:
        raise HTTPException(status_code=400, detail=f"Task is not completed. Current status: {video_task.status}")

    # VideoResult에서 해당 task_id와 video_type을 사용해 결과 파일 경로 조회
    result_file = db_crud.get_video_result_by_task_id_and_type(video_task.id, video_type)
    
    if not result_file or not os.path.exists(result_file):
        raise HTTPException(status_code=404, detail="Result file not found.")
    
    print(f"result_file: {result_file}", flush=True)

    # 파일 이름 설정: 'segments'는 ZIP, 그 외는 MP4
    if video_type in ['segments']:
        filename = f"download-{video_type}.zip"
    else:
        filename = f"download-{video_type}.mp4"

    # 비디오 파일 반환
    return FileResponse(result_file, filename=filename)


@app.post("/stop_task/{req_id}")
async def stop_task(req_id: str):
    """특정 req_id 작업 중지"""
    result = db_crud.cancel_video_task(req_id)
    print(f"stop_task result: {result}, req_id:{req_id}")
    return result
