from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from celery_app import celery_app
from app.models import VideoProcessInput
import uuid
import logging
import os
from urllib.parse import quote
from app.utils import set_stop_flag
import zipfile
from datetime import datetime, timezone

app = FastAPI()

# 로그 파일 설정
log_filename = "/base/logs/uvicorn/uvicorn.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s] %(message)s",
)

# POST /process_youtube_video: 유튜브 링크 기반 비디오 처리
@app.post("/process_youtube_video/")
async def youtube_video(input_data: VideoProcessInput):
    youtube_url = input_data.youtube_url
    start_time = input_data.start_time
    end_time = input_data.end_time
    
    # 받은 데이터를 출력 (추후 처리 로직 추가 가능)
    print(f"Received YouTube URL: {youtube_url}")
    print(f"Start Time: {start_time}")
    print(f"End Time: {end_time}")
    
    # Celery 작업 호출 시, input_data를 그대로 전달
    task = celery_app.send_task('app.tasks.process_youtube_video', args=[input_data.model_dump()])  # dict로 변환하여 전달
    
    # task ID 반환
    return {"message": "Video processing started", "task_id": task.id}

# POST /process_video: 비디오 파일 업로드 및 처리
@app.post("/process_video/")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # 1. 고유 작업 ID 생성
    print(f"upload video filename:{file.filename}")
    # 2. 파일을 서버에 저장
    video_path = os.path.join("/base/app/temp", f"{file.filename}")
    print(f"upload video video_path:{video_path}")
    with open(video_path, "wb") as f:
        f.write(await file.read())
    
    # 3. Celery 작업으로 파일 처리 요청
    task = celery_app.send_task('app.tasks.process_uploaded_video', args=[video_path])
    
    # 4. Celery 작업 ID 반환
    return {"message": "Video file uploaded and processing started", "task_id": task.id}

# GET /check_status/{task_id}: 작업 상태 확인
@app.get("/check_status/{task_id}")
async def check_status(task_id: str):
    task = celery_app.AsyncResult(task_id)
    
    print(f"check_status, task_id:{task_id}")
    print(f"check_status state, state:{task.state}")
    
    # 상태에 따른 처리
    if task.state == "PENDING":
        # 작업이 대기 중인 경우
        response = {"task_id": task_id, "status": "PENDING", "message": "The task is still pending."}
    elif task.state == "STARTED":
        proc_start_time = task.info.get('proc_start_time')
        print(f"proc_start_time:{proc_start_time}")
        if proc_start_time:
            # 시작 시간과 현재 시간 차이 계산
            elapsed_time = (datetime.now(timezone.utc) - proc_start_time).total_seconds()
            response = {"task_id": task_id, "status": "STARTED", "elapsed_time": f"{int(elapsed_time)}초"}
        else:
            response = {"task_id": task_id, "status": "STARTED", "message": "The task is currently being processed."}
    elif task.state == "SUCCESS":
        # 작업이 성공적으로 완료된 경우
        result = task.result  # 작업 결과 (process_video_task의 리턴값)
        proc_start_time = result.get('proc_start_time')
        proc_end_time = result.get('proc_end_time')

        if proc_start_time and proc_end_time:
            # 시작 시간과 종료 시간 차이 계산
            total_time = (proc_end_time - proc_start_time).total_seconds()
            response = {
                "task_id": task_id,
                "status": "SUCCESS",
                "message": "The task has been completed successfully.",
                "total_time": f"{int(total_time)}초"
            }
        else:
            response = {"task_id": task_id, "status": "SUCCESS", "message": "The task has been completed successfully."}
    elif task.state == "FAILURE":
        # 작업이 실패한 경우
        response = {"task_id": task_id, "status": "FAILURE", "message": str(task.info)}
    else:
        # 그 외 상태 처리
        response = {"task_id": task_id, "status": task.state, "message": "Unknown task state."}
    
    return response

# GET /download_video/{task_id}: 작업 완료된 비디오 파일 다운로드
@app.get("/download_video/{task_id}")
async def download_video(task_id: str, video_type: str = "full"):
    """
    task_id로 비디오 처리 상태를 확인한 후, 성공한 경우 결과 비디오 파일을 제공하는 엔드포인트.
    
    Parameters:
    - task_id: Celery 작업의 task_id
    - video_type: "output_video_1" 또는 "output_video_2" (기본값: "output_video_1")
    
    """
    
    # Celery 작업 상태 확인
    task = celery_app.AsyncResult(task_id)
    
    if task.state == "SUCCESS":
        result = task.result  # 작업 결과
        full_video_path = result.get("full_video_path")
        playing_video_path = result.get("playing_video_path")
        highlight_video_paths = result.get("highlight_video_paths")
        segment_zip_path = result.get("segment_zip_path")
        
        print(f"full_video_path:{full_video_path}, playing_video_path:{playing_video_path}, highlight_video_paths:{highlight_video_paths}, segment_zip_path:{segment_zip_path}")
        
        # video_type에 따라 파일 경로 설정
        if video_type == "full":
            result_file = full_video_path
            filename = 'download-full.mp4'
        elif video_type == "playing":
            result_file = playing_video_path
            filename = 'download-playing.mp4'
        elif video_type == "highlight":
            result_file = highlight_video_paths[0]
            filename = 'download-highlight.mp4'
        elif video_type == "highlights":
            base_path = os.path.dirname(highlight_video_paths[0])
            highlight_zip_path = os.path.join(base_path, "highlight_videos.zip")
            
            # ZIP 파일 생성
            with zipfile.ZipFile(highlight_zip_path, 'w') as zipf:
                for highlight_video in highlight_video_paths:
                    zipf.write(highlight_video, os.path.basename(highlight_video))  # 각 하이라이트 비디오를 압축
            
            result_file = highlight_zip_path
            filename = 'download-highlights.zip'
        elif video_type == "segments":
            result_file = segment_zip_path
            filename = 'download-segment.zip'
            
        print(f"result_file:{result_file}, filename:{filename}")
        
        # 해당 비디오 파일이 존재하는지 확인
        if not os.path.exists(result_file):
            return {"error": "Result file not found."}
        
        # 비디오 파일 반환
        return FileResponse(
            result_file, 
            filename=filename
        )
    
    elif task.state == "PENDING":
        return {"error": "The task is still pending or processing has not started yet."}
    elif task.state == "FAILURE":
        return {"error": "The task has failed."}
    else:
        return {"error": "The task is not completed yet."}

@app.post("/stop_task/{task_id}")
async def stop_task(task_id: str):
    """특정 task_id 작업 중지 요청"""
    set_stop_flag(task_id)
    return {"message": f"Task {task_id} has been requested to stop."}
