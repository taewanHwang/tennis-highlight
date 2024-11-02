import gradio as gr
import requests
import os
import time
import config
import db_crud

# 서버 URL 설정
process_upload_video_url = "http://0.0.0.0:9001/process_upload_video/"
process_youtube_video_url = "http://0.0.0.0:9001/process_youtube_video/"
check_status_url = "http://0.0.0.0:9001/check_status/"
download_url = "http://0.0.0.0:9001/download_video/"
stop_task_url = "http://0.0.0.0:9001/stop_task/"


# 비디오 업로드 함수
def process_upload_video(file, process_options, username):
    video_path = file.name
    with open(video_path, 'rb') as f:
        files = {'file': f}
        data = {
            'process_options': process_options,
            'username': username
            }
        response = requests.post(process_upload_video_url, files=files, data=data)
    if response.status_code == 200:
        response_data = response.json()
        return response_data.get("task_id")
    else:
        return None

# 유튜브 비디오 처리 함수
def process_youtube_video(youtube_link, start_time, end_time, process_options, username):
    print(f"process_youtube_video process_options:{process_options}")
    input_data = {
        "youtube_url": youtube_link,
        "start_time": start_time,
        "end_time": end_time,
        "process_options":process_options,
        "username":username
    }
    response = requests.post(process_youtube_video_url, json=input_data)
    if response.status_code == 200:
        response_data = response.json()
        return response_data.get("task_id")
    else:
        return None

# 작업 중지 함수
def stop_task(task_id):
    if task_id:
        stop_task_url_full = f"{stop_task_url}{task_id}"
        response = requests.post(stop_task_url_full)
        if response.status_code == 200:
            return f"Request {task_id} has been stopped."
        else:
            return f"Failed to stop task {task_id}. Status code: {response.status_code}"
    return "No task to stop."

# 상태 체크 함수
def check_status(task_id):
    if task_id:
        status_url = f"{check_status_url}{task_id}"
        
        retries = 3
        for attempt in range(retries):
            try:
                response_get = requests.get(status_url)
                if response_get.status_code == 200:
                    status_result = response_get.json()
                    print(f"status_result: {status_result}", flush=True)
                    
                    if status_result['status'] == "complete":
                        return status_result["message"], gr.update(interactive=True)
                    else:
                        return status_result["message"], gr.update(interactive=False)
                else:
                    print(f"Attempt {attempt+1} failed: {response_get.status_code}")
            except Exception as e:
                print(f"Attempt {attempt+1} failed with error: {e}")
            
            time.sleep(3)  # 1초 기다림

        return "Failed to get status after 3 attempts.", gr.update(interactive=False)
    
    return "No Task ID provided.", gr.update(interactive=False)


# 비디오 다운로드 함수
def download_video(task_id, video_type):
    if task_id:
        # 서버에서 비디오를 다운로드할 URL 생성
        download_video_url = f"{download_url}{task_id}?video_type={video_type}"
        response_download = requests.get(download_video_url, stream=True)

        if response_download.status_code == 200:
            # Content-Disposition 헤더에서 파일명 추출
            content_disposition = response_download.headers.get('Content-Disposition')
            print(f"content_disposition:{content_disposition}",flush=True)
            filename = content_disposition.split('filename=')[-1].strip('\"')
            print(f"filename:{filename}",flush=True)

            # 서버에서 전달된 파일명을 사용하여 파일 저장
            file_path = os.path.join(os.getcwd(), filename)
            print(f"file_path:{file_path}",flush=True)
            with open(file_path, 'wb') as f:
                for chunk in response_download.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            
            if os.path.exists(file_path):
                print(f"file exist:{file_path}",flush=True)
                return file_path
            else:
                return "Error: Downloaded file not found."
        else:
            return f"Failed to download file. Status code: {response_download.status_code}"
    return "No task to download video from."


def auth(username, password):
    is_authenticated = db_crud.authenticate_user(username, password)
    print(f"Authentication attempted for user: {username}, result: {is_authenticated}", flush=True)

    return is_authenticated


def show_task_list():
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

def show_task_detail():
    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

def show_new_task():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

def on_row_select(task_state, evt: gr.SelectData):
    
    status = evt.row_value[1]
    
    if status == config.TASK_STAT_COMPLETED:
        tl, td, nt = show_task_detail()

        task_id = evt.row_value[0]
        task_state["detail_task_id"] = task_id
        
        button_state = gr.update(interactive=True)
    else:
        gr.Warning("Select completed task")
        task_state["detail_task_id"] = None
        tl, td, nt = show_task_list()     
           
        button_state = gr.update(interactive=False)
    return tl, td, nt, button_state, button_state, button_state, button_state, task_state


def init_tasks_func(request: gr.Request):
    auth_state, task_table = update_tasks_func(request)
    username = auth_state.get("username", "Guest")
    user_info_message = (
        f"### 환영합니다. {username}!\n"
        f"작업 목록을 새로고침하려면 **'Update Tasks'** 버튼을, 새 작업을 시작하려면 **'New Task'** 버튼을 눌러주세요."
    )
    return auth_state, task_table, user_info_message

def update_tasks_func(request: gr.Request):
    auth_state = auth_state = {"username": request.username} 
    task_table = db_crud.get_user_tasks(auth_state)
    return auth_state, task_table

def update_ui(input_method):
    inter_false = gr.update(visible=False, interactive=False)
    inter_true = gr.update(visible=True, interactive=True)
    if input_method == "File Upload":
        return (inter_true,  # 파일 업로드 필드를 활성화
                inter_false,  # 유튜브 링크 비활성화
                inter_false,  # 시작 시간 비활성화
                inter_false)  # 끝 시간 비활성화
    elif input_method == "YouTube Link":
        return (inter_false,  # 파일 업로드 비활성화
                inter_true,  # 유튜브 링크 필드를 활성화
                inter_true,  # 시작 시간 필드를 활성화
                inter_true)  # 끝 시간 필드를 활성화

# 파일 업로드 및 유튜브 링크 처리 함수
def start_processing(upload_file, youtube_link, start_time, end_time, input_method, process_options, task_state, auth_state):
    username = auth_state.get("username")  # auth_state에서 username 가져오기
    
    # process_options가 비어있는지 확인
    if not process_options:
        gr.Warning("No processing options provided")
        return "No processing options provided.", task_state

    print(f"start_processing process_options:{process_options}, username:{username}, task_state:{task_state}, auth_state:{auth_state}", flush=True)
    if input_method == "File Upload" and upload_file is not None:
        task_id = process_upload_video(upload_file, process_options, username)
        task_state["created_task_id"] = task_id
    elif input_method == "YouTube Link" and youtube_link is not None:
        task_id = process_youtube_video(youtube_link, start_time, end_time, process_options, username)
        task_state["created_task_id"] = task_id
    else:
        gr.Warning("No valid input provided")
        return "No valid input provided.", task_state
        
    if task_id:
        gr.Info("Success to start processing")
        return "Success to start processing", task_state
    else:
        gr.Warning("Failed to start processing")
        return "Failed to start processing.", task_state

# 작업 중지 처리 함수
def stop_processing(task_state):
    task_id = task_state["created_task_id"]
    if task_id:
        stop_response =  stop_task(task_id)  # 작업 중지 요청
        gr.Info(stop_response)
        return stop_response
    
    gr.Warning("No task to stop")
    return "No task to stop."

# 리프레시 버튼으로 상태 갱신
def refresh_status(task_state):
    task_id = task_state["created_task_id"]
    if task_id:
        status, _ = check_status(task_id)
        gr.Info(status)
        return status
    
    gr.Warning("No task in progress")
    return "No task in progress."

# 다운로드 버튼으로 전체 비디오 다운로드
def start_download_full(task_state):
    task_id = task_state["detail_task_id"]
    if task_id:
        print("start_download_full",flush=True)
        return download_video(task_id, "full")  # 전체 비디오 다운로드
    return "No task to download."

# 다운로드 버튼으로 플레이 중인 비디오 다운로드
def start_download_playing(task_state):
    task_id = task_state["detail_task_id"]
    if task_id:
        print("start_download_playing",flush=True)
        return download_video(task_id, "playing")  # 플레이 중인 비디오 다운로드
    return "No task to download."

# 다운로드 버튼으로 하이라이트 복수 비디오 다운로드
def start_download_highlights(task_state):
    task_id = task_state["detail_task_id"]
    if task_id:
        print("start_download_highlights",flush=True)
        return download_video(task_id, "highlights")  # 하이트 비디오 다운로드
    return "No task to download."

# 다운로드 버튼으로 세그먼트 비디오 다운로드
def start_download_segments(task_state):
    task_id = task_state["detail_task_id"]
    if task_id:
        print("start_download_segments",flush=True)
        return download_video(task_id, "segments")  # 하이트 비디오 다운로드
    return "No task to download."
