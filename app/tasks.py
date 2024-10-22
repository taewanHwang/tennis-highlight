from celery import shared_task
from app.utils import process_video, download_youtube_video, set_stop_flag, is_task_stopped
from datetime import datetime, timezone


@shared_task(bind=True, name="app.tasks.process_youtube_video", track_started=True)
def process_youtube_video(self, input_data):
    """
    유튜브 비디오를 다운로드하고, 비디오 처리 및 저장.
    """
    proc_start_time = datetime.now(timezone.utc)
    self.update_state(state='STARTED', meta={'proc_start_time': proc_start_time})

    # 유튜브 링크와 시작/종료 시간 추출
    youtube_url = input_data['youtube_url']
    start_time = input_data['start_time']
    end_time = input_data['end_time']
    
    print(f"process_youtube_video 실행: youtube_url: {youtube_url}, start_time: {start_time}, end_time: {end_time}")

    print(f'task_id:{self.request.id}, is_stoppped:{is_task_stopped(self.request.id)}')
    if is_task_stopped(self.request.id):
        self.update_state(state='REVOKED', meta={"info": "Task was stopped by user."})
        return {"message": "Task was stopped by user"}

    # # 1. 유튜브 비디오 다운로드
    video_path = download_youtube_video(youtube_url, start_time, end_time)
    (full_video_path, playing_video_path, highlight_video_paths, segment_zip_path) = process_video(self, video_path)

    # 6. 결과 반환
    return {
        "message": "Video processing completed",
        "full_video_path": full_video_path,
        "playing_video_path": playing_video_path,
        "highlight_video_paths": highlight_video_paths,
        "segment_zip_path": segment_zip_path,
        'proc_start_time': proc_start_time,
        'proc_end_time': datetime.now(timezone.utc), 
    }


@shared_task(bind=True, name="app.tasks.process_uploaded_video", track_started=True)
def process_uploaded_video(self, video_path):
    self.update_state(state='STARTED')  # 작업 시작 시 상태 업데이트
    """
    업로드된 비디오 파일을 처리하고, 결과를 저장.
    """
    proc_start_time = datetime.now(timezone.utc)
    self.update_state(state='STARTED', meta={'proc_start_time': proc_start_time})

    (full_video_path, playing_video_path, highlight_video_paths, segment_zip_path) = process_video(self, video_path)

    # 6. 결과 반환
    return {
        "message": "Video processing completed",
        "full_video_path": full_video_path,
        "playing_video_path": playing_video_path,
        "highlight_video_paths": highlight_video_paths,
        "segment_zip_path": segment_zip_path,
        'proc_start_time': proc_start_time,
        'proc_end_time': datetime.now(timezone.utc), 
    }
    

@shared_task(bind=True, name="app.tasks.sto_task", track_started=True)
def stop_process(self, input_data):
    """
    유튜브 비디오를 다운로드하고, 비디오 처리 및 저장.
    """
    task_id = self.request.id
    ### task id에 해당하는 redis 작업을 중지, 중지되었는지 확인하고 true, false 반환
    set_stop_flag(task_id)
    return {"status": "stopping", "task_id": task_id}
