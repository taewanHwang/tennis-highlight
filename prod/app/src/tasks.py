from celery import shared_task
from utils import process_video, download_youtube_video
import db_crud

@shared_task(bind=True, name="app.src.tasks.process_youtube_video", track_started=True)
def process_youtube_video(self, input_data):
    """
    유튜브 비디오를 다운로드하고, 비디오 처리 및 저장.
    """
    try:
        # 유튜브 링크와 시작/종료 시간 추출
        youtube_url = input_data['youtube_url']
        start_time = input_data['start_time']
        end_time = input_data['end_time']
        process_options = input_data['process_options']
        username = input_data['username']
        req_id = self.request.id  # req ID 가져오기

        print(f"process_youtube_video 실행: youtube_url: {youtube_url}, start_time: {start_time}, end_time: {end_time}, process_options:{process_options}, username:{username}, req_id: {req_id}")
        
        # 1. DB에 video task 생성
        video_task = db_crud.create_video_task(username, req_id, youtube_url, process_options)

        if db_crud.is_task_cancelled(video_task.id):
            return {"message": "Task was stopped by user"}

        # 2. 유튜브 비디오 다운로드
        video_path = download_youtube_video(youtube_url, start_time, end_time, username)
        
        if db_crud.is_task_cancelled(video_task.id):
            return {"message": "Task was stopped by user"}
        db_crud.update_task_status_to_downloaded(video_task.id)
            
        # 3. 비디오 영상 처리
        full_video_path, playing_video_path, highlight_zip_path, segment_zip_path = process_video(self, video_path, process_options, username)
        
        if db_crud.is_task_cancelled(video_task.id):
            return {"message": "Task was stopped by user"}
        
        # 비디오 처리 결과를 VideoResult에 저장
        if full_video_path:  # full_video_path가 None이 아닌 경우
            db_crud.create_video_result(video_task.id, full_video_path, "full")

        if playing_video_path:  # playing_video_path가 None이 아닌 경우
            db_crud.create_video_result(video_task.id, playing_video_path, "playing")

        if highlight_zip_path:  # highlight_zip_path가 None이 아닌 경우
            db_crud.create_video_result(video_task.id, highlight_zip_path, "highlights")

        if segment_zip_path:  # segment_zip_path가 None이 아닌 경우
            db_crud.create_video_result(video_task.id, segment_zip_path, "segments")

        db_crud.update_task_status_to_complete(video_task.id)
        
    except Exception as e:
        print(f"Error in process_youtube_video: {e}", flush=True)
        
        # 에러 발생 시 해당 작업의 상태를 "error"로 업데이트
        if video_task:
            db_crud.update_task_status_to_error(video_task.id)
        
        # 에러 메시지를 반환하거나 Celery에 기록
        raise e 

@shared_task(bind=True, name="app.src.tasks.process_uploaded_video", track_started=True)
def process_uploaded_video(self, video_path, process_options, username):
    """
    업로드된 비디오 파일을 처리하고, 결과를 저장.
    """
    try:
        req_id = self.request.id  # req ID 가져오기
        print(f"process_uploaded_video 실행: video_path: {video_path}, process_options: {process_options}, username: {username}, req_id:{req_id}")
        
        # 1. DB에 video task 생성
        video_task = db_crud.create_video_task(username, req_id, video_path, process_options)

        # 2. 비디오 영상 처리
        full_video_path, playing_video_path, highlight_zip_path, segment_zip_path = process_video(self, video_path, process_options, username)
        if db_crud.is_task_cancelled(video_task.id):
            return {"message": "Task was stopped by user"}
        
        # 비디오 처리 결과를 VideoResult에 저장
        if full_video_path:  # full_video_path가 None이 아닌 경우
            db_crud.create_video_result(video_task.id, full_video_path, "full")

        if playing_video_path:  # playing_video_path가 None이 아닌 경우
            db_crud.create_video_result(video_task.id, playing_video_path, "playing")

        if highlight_zip_path:  # highlight_zip_path가 None이 아닌 경우
            db_crud.create_video_result(video_task.id, highlight_zip_path, "highlights")

        if segment_zip_path:  # segment_zip_path가 None이 아닌 경우
            db_crud.create_video_result(video_task.id, segment_zip_path, "segments")

        db_crud.update_task_status_to_complete(video_task.id)
    except Exception as e:
        print(f"Error in process_uploaded_video: {e}", flush=True)
        
        # 에러 발생 시 해당 작업의 상태를 "error"로 업데이트
        if video_task:
            db_crud.update_task_status_to_error(video_task.id)
        
        # 에러 메시지를 반환하거나 Celery에 기록
        raise e 
