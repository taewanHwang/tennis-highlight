import gradio as gr
import requests
import os
from app.utils import load_users

# 서버 URL 설정
process_video_url = "http://0.0.0.0:9001/process_video/"
process_youtube_video_url = "http://0.0.0.0:9001/process_youtube_video/"
check_status_url = "http://0.0.0.0:9001/check_status/"
download_url = "http://0.0.0.0:9001/download_video/"
stop_task_url = "http://0.0.0.0:9001/stop_task/"

# 비디오 업로드 함수
def process_video(file):
    video_path = file.name
    with open(video_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(process_video_url, files=files)
    if response.status_code == 200:
        response_data = response.json()
        return response_data.get("task_id")
    else:
        return None

# 유튜브 비디오 처리 함수
def process_youtube_video(youtube_link, start_time, end_time):
    input_data = {
        "youtube_url": youtube_link,
        "start_time": start_time,
        "end_time": end_time
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
            return f"Task {task_id} has been stopped."
        else:
            return f"Failed to stop task {task_id}. Status code: {response.status_code}"
    return "No task to stop."

# 상태 체크 함수
def check_status(task_id):
    if task_id:
        status_url = f"{check_status_url}{task_id}"
        response_get = requests.get(status_url)
        if response_get.status_code == 200:
            status_result = response_get.json()
            if status_result['status'] == "STARTED":
                result_text = f"{status_result['elapsed_time']} 동안 처리중입니다."
                return result_text, gr.update(interactive=False)  # 다운로드 비활성화
            elif status_result['status'] == "PENDING":
                return "대기중입니다.", gr.update(interactive=False)  # 다운로드 비활성화
            elif status_result['status'] == "SUCCESS":
                result_text = f"완료 되었습니다. 다운로드 가능합니다. {status_result['total_time']} 소요되었습니다."
                return result_text, gr.update(interactive=True)  # 다운로드 활성화
            else:
                return f"Current Status: {status_result['status']}", gr.update(interactive=False)  # 다운로드 비활성화
        else:
            return "Failed to get status.", gr.update(interactive=False)  # 다운로드 비활성화
    return "No Task ID provided.", gr.update(interactive=False)  # 다운로드 비활성화


# 비디오 다운로드 함수
def download_video(task_id, video_type):
    if task_id:
        # 서버에서 비디오를 다운로드할 URL 생성
        download_video_url = f"{download_url}{task_id}?video_type={video_type}"
        response_download = requests.get(download_video_url, stream=True)

        if response_download.status_code == 200:
            # Content-Disposition 헤더에서 파일명 추출
            content_disposition = response_download.headers.get('Content-Disposition')
            if content_disposition:
                # 파일명 추출 (Content-Disposition: attachment; filename="파일명")
                filename = content_disposition.split('filename=')[-1].strip('\"')
            else:
                # 파일명을 헤더에서 찾지 못한 경우 기본 파일명 설정
                filename = f"{task_id}_{video_type}.mp4"

            # 서버에서 전달된 파일명을 사용하여 파일 저장
            video_file_path = os.path.join(os.getcwd(), filename)
            with open(video_file_path, 'wb') as f:
                for chunk in response_download.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            
            if os.path.exists(video_file_path):
                return video_file_path
            else:
                return "Error: Downloaded video file not found."
        else:
            return f"Failed to download video. Status code: {response_download.status_code}"
    return "No task to download video from."

# Gradio UI 구성
with gr.Blocks() as demo:
    
    # 입력 방식 선택 (파일 업로드 또는 유튜브 링크 입력)
    input_method = gr.Dropdown(["File Upload", "YouTube Link"], label="Choose input method", value="File Upload")
    
    # 파일 업로드 관련 필드
    upload_file = gr.File(label="Upload Video File", visible=True)
    
    # 유튜브 링크 및 시간 입력 관련 필드 (처음에는 보이지 않음)
    youtube_link = gr.Textbox(label="YouTube Video Link", visible=False)
    start_time = gr.Textbox(label="Start Time (HH:MM:SS)", value="00:00:00", visible=False)
    end_time = gr.Textbox(label="End Time (HH:MM:SS)", value="00:01:00", visible=False)

    process_button = gr.Button("Start Processing")
    stop_button = gr.Button("Stop Processing", interactive=True)  # 작업 중지 버튼 추가

    # 상태 확인 및 다운로드 버튼
    status_output = gr.Textbox(label="Status", lines=2)
    refresh_button = gr.Button("Refresh Status")
    download_full_button = gr.Button("Download Full Video", interactive=False)
    download_playing_button = gr.Button("Download Playing Video", interactive=False)
    download_highlight_button = gr.Button("Download Highlight Video", interactive=False)
    download_highlights_button = gr.Button("Download Highlights Video", interactive=False)
    download_segments_button = gr.Button("Download Segments Video", interactive=False)
    download_output = gr.File(label="Download Processed Video", interactive=False)
    
    # task_id 저장을 위한 상태
    task_state = gr.State(value="")
    
    # 입력 방식에 따라 화면 요소를 동적으로 업데이트하는 함수
    def update_ui(input_method):
        if input_method == "File Upload":
            return (gr.update(visible=True, interactive=True),  # 파일 업로드 필드를 활성화
                    gr.update(visible=False, interactive=False),  # 유튜브 링크 비활성화
                    gr.update(visible=False, interactive=False),  # 시작 시간 비활성화
                    gr.update(visible=False, interactive=False))  # 끝 시간 비활성화
        elif input_method == "YouTube Link":
            return (gr.update(visible=False, interactive=False),  # 파일 업로드 비활성화
                    gr.update(visible=True, interactive=True),  # 유튜브 링크 필드를 활성화
                    gr.update(visible=True, interactive=True),  # 시작 시간 필드를 활성화
                    gr.update(visible=True, interactive=True))  # 끝 시간 필드를 활성화

    # 파일 업로드 및 유튜브 링크 처리 함수
    def start_processing(upload_file, youtube_link, start_time, end_time, input_method, state):
        if input_method == "File Upload" and upload_file is not None:
            task_id = process_video(upload_file)
        elif input_method == "YouTube Link" and youtube_link is not None:
            task_id = process_youtube_video(youtube_link, start_time, end_time)
        else:
            return "No valid input provided.", state, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
        
        if task_id:
            state = task_id  # task_id를 상태로 저장
            status, _ = check_status(task_id)
            return status, state, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False) , gr.update(interactive=False) 
        return "Failed to start processing.", state, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
    
    # 작업 중지 처리 함수
    def stop_processing(state):
        if state:
            return stop_task(state)  # 작업 중지 요청
        return "No task to stop."

    # 리프레시 버튼으로 상태 갱신
    def refresh_status(state):
        if state:
            status, button_state = check_status(state)
            return status, button_state, button_state, button_state, button_state, button_state
        return "No task in progress.", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)

    # 다운로드 버튼으로 전체 비디오 다운로드
    def start_download_full(state):
        if state:
            return download_video(state, "full")  # 전체 비디오 다운로드
        return "No task to download."

    # 다운로드 버튼으로 플레이 중인 비디오 다운로드
    def start_download_playing(state):
        if state:
            return download_video(state, "playing")  # 플레이 중인 비디오 다운로드
        return "No task to download."

    # 다운로드 버튼으로 하이라이트 중인 비디오 다운로드
    def start_download_highlight(state):
        if state:
            return download_video(state, "highlight")  # 하이트 비디오 다운로드
        return "No task to download."

    # 다운로드 버튼으로 하이라이트 복수 비디오 다운로드
    def start_download_highlights(state):
        if state:
            return download_video(state, "highlights")  # 하이트 비디오 다운로드
        return "No task to download."

    # 다운로드 버튼으로 세그먼트 비디오 다운로드
    def start_download_segments(state):
        if state:
            return download_video(state, "segments")  # 하이트 비디오 다운로드
        return "No task to download."

    # 입력 방식 변경에 따른 UI 업데이트
    input_method.change(update_ui, inputs=input_method, outputs=[upload_file, youtube_link, start_time, end_time])

    # 이벤트 핸들러 설정
    process_button.click(start_processing, inputs=[upload_file, youtube_link, start_time, end_time, input_method, task_state], 
                         outputs=[status_output, task_state, download_full_button, download_playing_button, download_highlight_button, download_highlights_button, download_segments_button])
    
    
    stop_button.click(stop_processing, inputs=task_state, outputs=[status_output]) 
    refresh_button.click(refresh_status, inputs=task_state, outputs=[status_output, download_full_button, download_playing_button, download_highlight_button, download_highlights_button, download_segments_button])
    download_full_button.click(start_download_full, inputs=task_state, outputs=download_output)
    download_playing_button.click(start_download_playing, inputs=task_state, outputs=download_output)
    download_highlight_button.click(start_download_highlight, inputs=task_state, outputs=download_output)
    download_highlights_button.click(start_download_highlights, inputs=task_state, outputs=download_output)
    download_segments_button.click(start_download_segments, inputs=task_state, outputs=download_output)


# 사용자 정보 파일에서 읽어와서 인증 정보로 설정
auth_users = load_users()

# Gradio 웹 인터페이스 실행
demo.launch(auth=auth_users, share=False, server_port=9000, server_name="0.0.0.0")
