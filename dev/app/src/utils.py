import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from torchvision.io import read_video
import yt_dlp
import os
import re
from datetime import datetime
from torchvision.transforms import Compose, Lambda, Resize
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample, Normalize
import uuid
import config as config
import cv2
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import time
import zipfile
from datetime import datetime, timezone, timedelta
import re
from urllib.parse import urlparse

# UTC 시간을 KST로 변환하는 함수
def convert_to_kst(utc_datetime):
    kst_timezone = timezone(timedelta(hours=9))  # 한국 시간대는 UTC+9
    return utc_datetime.astimezone(kst_timezone)

def format_datetime_short(dt):
    """
    주어진 datetime 객체를 'MM/DD HH:MM' 형식의 문자열로 변환합니다.
    """
    return dt.strftime('%m/%d %H:%M')

def validate_upload_video_input(file, process_options, username):
    
    # 파일 확장자가 유효한 비디오 형식인지 확인
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    _, file_extension = os.path.splitext(file.filename.lower())
    if file_extension not in allowed_extensions:
        raise ValueError(f"Invalid file format: {file_extension}. Allowed formats are: {', '.join(allowed_extensions)}")

    # process_options가 최소한 하나 이상의 요소를 가지고 있는지 확인
    if not process_options or len(process_options) < 1:
        raise ValueError("Process options must contain at least one item.")

    # username이 비어있지 않은지 확인
    if not username or username.strip() == "":
        raise ValueError("Username cannot be empty.")

def validate_youtube_video_input(input_data):
    # 유효한 YouTube URL 확인
    youtube_url = input_data.youtube_url
    if not is_valid_youtube_url(youtube_url):
        raise ValueError("Invalid YouTube URL.")

    # HH:MM:SS 포맷 확인
    start_time = input_data.start_time
    end_time = input_data.end_time
    time_format = r"^\d{2}:\d{2}:\d{2}$"
    if not re.match(time_format, start_time) or not re.match(time_format, end_time):
        raise ValueError("Start time and end time must be in HH:MM:SS format.")

    # end_time이 start_time보다 뒤에 있는지 확인
    if not is_end_time_after_start(start_time, end_time):
        raise ValueError("End time must be after start time.")

    # process_options가 최소한 하나 이상의 요소를 가지고 있는지 확인
    process_options = input_data.process_options
    if len(process_options) < 1:
        raise ValueError("Process options must contain at least one item.")

    # username이 비어있지 않은지 확인
    if not input_data.username or input_data.username.strip() == "":
        raise ValueError("Username cannot be empty.")


def is_valid_youtube_url(url):
    try:
        parsed_url = urlparse(url)
        return parsed_url.netloc in ["www.youtube.com", "youtube.com", "youtu.be"]
    except:
        return False

def is_end_time_after_start(start_time, end_time):
    # HH:MM:SS 포맷을 초 단위로 변환 후 비교
    def time_to_seconds(t):
        h, m, s = map(int, t.split(":"))
        return h * 3600 + m * 60 + s
    
    return time_to_seconds(end_time) > time_to_seconds(start_time)

def format_filename(title, max_length=50):
    # 파일명에 적합하지 않은 문자 제거 (파일명으로 사용할 수 없는 문자)
    title = re.sub(r'[\\/*?:"<>|]', "", title)
    # 길이가 50자를 넘으면 자르기
    return title[:max_length]

def download_youtube_video(youtube_link, start_time, end_time, username):
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 유튜브 제목 가져오기
    with yt_dlp.YoutubeDL() as ydl:
        info_dict = ydl.extract_info(youtube_link, download=False)
        video_title = info_dict.get('title', 'Unknown Title')

    # 제목 포맷팅 (50자 제한 및 특수문자 제거)
    formatted_title = format_filename(video_title)

    # 파일명 생성 (유튜브 제목 + 현재 시간)
    output_filename = f"{formatted_title}_{current_time}.mp4"

    # username 폴더 경로 생성
    video_temp_folder = os.path.join(config.VIDEO_DATA_FOLDER, username, "temp")

    # 폴더가 없으면 생성
    if not os.path.exists(video_temp_folder):
        os.makedirs(video_temp_folder, exist_ok=True)

    # 파일이 저장될 전체 경로
    output_path = os.path.join(video_temp_folder, output_filename)

    ydl_opts = {
        'outtmpl': output_path,
        'format': 'bestvideo+bestaudio/best',
        'postprocessor_args': ['-ss', start_time, '-to', end_time],
        'merge_output_format': 'mp4',
        'quiet': True,
        'no_warnings': True,
        'progress_hooks': [lambda d: None]  # 진행률 로그 비활성화
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_link])

    return output_path

def download_youtube_video_segment(youtube_link, start_time, end_time, username):
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Fetch video information without downloading
    with yt_dlp.YoutubeDL() as ydl:
        info_dict = ydl.extract_info(youtube_link, download=False)
        video_title = info_dict.get('title', 'Unknown Title')
    
    # Format the video title and paths
    formatted_title = "".join(c if c.isalnum() else "_" for c in video_title)[:50]
    temp_filename = f"{formatted_title}_{current_time}_temp.mp4"
    output_filename = f"{formatted_title}_{current_time}_segment.mp4"
    video_temp_folder = os.path.join("/path/to/video/storage", username, "temp")

    # Create temp folder if it doesn't exist
    os.makedirs(video_temp_folder, exist_ok=True)
    
    temp_path = os.path.join(video_temp_folder, temp_filename)
    output_path = os.path.join(video_temp_folder, output_filename)

    # yt-dlp options for downloading
    ydl_opts = {
        'outtmpl': temp_path,
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4'
    }

    # Download the video fully
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_link])

    # Use ffmpeg to extract the desired segment
    ffmpeg_command = [
        'ffmpeg', '-i', temp_path, '-ss', start_time, '-to', end_time,
        '-c', 'copy', output_path
    ]

    # Run ffmpeg command
    subprocess.run(ffmpeg_command, check=True)

    # Clean up the temp full video
    os.remove(temp_path)

    return output_path



def process_video_for_predictions(video_file_path, model, val_transform, chunk_size=16, target_fps=6, postprocess=True):
    cap = cv2.VideoCapture(video_file_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)  # 원본 fps 가져오기
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 비디오의 총 프레임 수 가져오기

    print(f"video_file_path: {video_file_path}")
    print(f"Total number of frames: {total_frames}")
    print(f"original_fps: {original_fps}")

    frame_index = 0
    predictions = []

    while cap.isOpened():
        # target_fps로 청크를 처리하고 예측 결과 받기
        prediction, frame_index = process_chunk_for_prediction(cap, model, val_transform, chunk_size, frame_index, target_fps)
        cur_time = frame_index/target_fps
        print(f"frame_index: {frame_index}, time:{cur_time}, prediction: {prediction}")

        if prediction is None:
            break  # 프레임이 충분하지 않으면 종료

        predicted_class, playing_prob, not_playing_prob = prediction
        predictions.append((frame_index / target_fps, predicted_class, playing_prob, not_playing_prob))

    cap.release()

    # 후처리 적용
    if postprocess:
        predictions = postprocess_predictions(predictions, config.POST_CONS_CHUNK_SIZE)

    print(f"predictions:{predictions}")
    print(f"original_fps:{original_fps}")

    # 예측 결과를 반환
    return predictions, original_fps

def postprocess_predictions(predictions, n):
    """
    연속 n개의 청크가 playing이고, 현재 청크가 not-playing인 경우 후처리하여 playing으로 변경.
    후처리가 적용된 이후, 새로운 playing 구간이 발생하면 후처리 상태를 초기화함.
    """
    play_streak = []  # 최근 n개의 청크 저장용
    processed_predictions = []
    postprocess_applied = False  # 후처리가 한 번 적용되었는지 확인하는 플래그

    for prediction in predictions:
        time_stamp, predicted_class, playing_prob, not_playing_prob = prediction

        # 최근 청크 기록 관리 (최대 n개 유지)
        play_streak.append((predicted_class, playing_prob, not_playing_prob))
        if len(play_streak) > n:
            play_streak.pop(0)

        # 후처리 상태 초기화: 새로운 playing 구간이 발생하면 초기화
        if predicted_class == 'playing':
            postprocess_applied = False

        # 후처리 조건 체크 (후처리가 이미 적용된 상태에서는 스킵)
        if len(play_streak) == n and not postprocess_applied:
            last_n_chunks = [chunk[0] for chunk in play_streak]  # 최근 n개 청크의 클래스만 가져옴
            if last_n_chunks[:-1] == ['playing'] * (n - 1) and last_n_chunks[-1] == 'not-playing':
                # 현재 청크가 not-playing이고 playing_prob가 낮을 경우 후처리
                if play_streak[-1][1] <= config.PLAY_THRESHOLD:
                    print(f"Postprocessing: Adjusting prediction at time {time_stamp}. Changing 'not-playing' to 'playing'.")
                    # 현재 청크를 playing으로 수정
                    processed_predictions.append((time_stamp, 'playing', 0.9, 0.1))
                    # play_streak에서도 수정
                    play_streak[-1] = ('playing', 0.9, 0.1)
                    # 후처리 적용 플래그 설정
                    postprocess_applied = True
                else:
                    processed_predictions.append(prediction)
            else:
                processed_predictions.append(prediction)
        else:
            processed_predictions.append(prediction)

    return processed_predictions

def compress_video_with_ffmpeg(input_path, output_path):
    command = [
        'ffmpeg',
        '-i', input_path,           # 입력 파일 경로
        '-vcodec', 'libx264',       # H.264 코덱 사용
        '-crf', '23',               # 품질 설정 (값을 높이면 더 압축, 낮추면 더 좋은 품질)
        '-preset', 'medium',        # ultrafast, superfast, veryfast, faster, fast, medium (기본값), slow, slower, veryslow
        '-y',                       # 기존 파일 덮어쓰기
        output_path                 # 출력 파일 경로
    ]

    try:
        # ffmpeg 명령 실행
        subprocess.run(command, check=True)
        print(f"Video successfully compressed and saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while compressing video: {e}")

def find_playing_segments(predictions, chunk_duration):
    playing_segments = []
    current_segment = None

    for idx, (chunk_end_time, label, playing_prob, _) in enumerate(predictions):
        chunk_start_time = chunk_end_time - chunk_duration  # 청크 시작 시간을 계산
        if label == 'playing':
            if current_segment is None:
                # 새로운 playing 세그먼트 시작
                current_segment = [chunk_start_time]  # 시작 시간을 기록 (청크 시작 시간)
        else:
            if current_segment is not None:
                # playing 세그먼트가 끝났으므로 종료 시간 기록
                current_segment.append(chunk_end_time)  # 끝 시간을 기록 (청크 끝 시간)
                playing_segments.append(current_segment)
                current_segment = None

    # 마지막 세그먼트가 끝나지 않았을 경우 종료 시간 추가
    if current_segment is not None:
        current_segment.append(predictions[-1][0])  # 마지막 청크의 끝 시간으로 설정
        playing_segments.append(current_segment)

    return playing_segments


def find_longest_segments(segments, n):
    if not segments:
        return []  # 세그먼트가 없으면 빈 리스트 반환

    # 각 세그먼트의 길이를 계산하여 긴 순서대로 정렬
    sorted_segments = sorted(segments, key=lambda seg: seg[1] - seg[0], reverse=True)

    # 최대 n개의 세그먼트를 반환
    return sorted_segments[:n]

def create_highlight_video(video_file_path, longest_segment, output_path):
    cap = cv2.VideoCapture(video_file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 원본 프레임 형식 유지

    # 비디오 라이터 초기화
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start_time, end_time = longest_segment
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    out.release()
    cap.release()

    print(f"Created highlight video: {output_path}")

def create_segment_videos(video_file_path, segments, base_dir):
    cap = cv2.VideoCapture(video_file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 원본 프레임 형식 유지

    segment_videos = []

    for idx, (start_time, end_time) in enumerate(segments):
        # 비디오 파일명 생성
        output_file = os.path.join(base_dir, f'temp_segment{idx + 1}.mp4')

        # 비디오 라이터 초기화
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        # 비디오 프레임 계산
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()
        segment_videos.append(output_file)
        print(f"Created video segment: {output_file}")

    cap.release()

    return segment_videos



def extract_video_using_prediction(video_path, temp_video_path, output_video_path, predictions, original_fps, extract_type):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    cvwriter = cv2.VideoWriter(temp_video_path, fourcc, original_fps, (width, height))
    if not cvwriter.isOpened():
        raise IOError(f"Error: Failed to open video writer for {temp_video_path}. Ensure that codec is supported.")

    previous_end_frame = 0
    for idx, (time_index, pred_label, playing_prob, not_playing_prob) in enumerate(predictions):
        start_frame = previous_end_frame
        end_frame = int(time_index * original_fps)
        print(f"start_frame:{start_frame}, end_frame:{end_frame}, time_index:{time_index}, playing_prob:{playing_prob}")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 프레임 읽고 처리
        frame_start_time = time.time()
        for frame_index in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            if extract_type == "full":
                frame_with_text = add_prediction_text_to_frame(frame, playing_prob, not_playing_prob, start_frame / original_fps, end_frame / original_fps)
                cvwriter.write(frame_with_text)

            if (extract_type == "playing") and (playing_prob > config.PLAY_THRESHOLD) :
                cvwriter.write(frame)  # 비디오2에 해당 프레임 저장
                
        previous_end_frame = end_frame

    cap.release()
    cvwriter.release()

    print(f"Full video with text saved at: {temp_video_path}")

    compress_video_with_ffmpeg(temp_video_path, output_video_path)  # 전체 비디오 압축
    os.remove(temp_video_path)

    print(f"Compressed and saved at: {temp_video_path}, {output_video_path}")
    
    return output_video_path

def extract_videos_using_prediction(predictions, video_file_path, highlight_temp_video_path, highlight_video_path, folder_path, segment_zip_path, highlight_zip_path):
    chunk_duration = predictions[0][0]  # 각 청크의 지속 시간

    # 1. 연속적인 playing 구간 찾기
    playing_segments = find_playing_segments(predictions, chunk_duration)
    print(f"Playing Segments: {playing_segments}")

    # 2. 가장 긴 세그먼트 찾기
    highlight_segments = find_longest_segments(playing_segments, n=config.HIGHTLIGHTS_NUM)
    print(f"Highlight Segment: {highlight_segments}")

    # 3-1. 연속적인 세그먼트에 해당하는 비디오 생성
    segment_videos = create_segment_videos(video_file_path, playing_segments, folder_path)
    print(f"segment_videos: {segment_videos}")
    segment_comp_videos = []  # 압축된 비디오 파일들을 저장할 리스트
    
    for segment_video in segment_videos:
        segment_comp_video = segment_video.replace("temp_", "")
        compress_video_with_ffmpeg(segment_video, segment_comp_video) 
        segment_comp_videos.append(segment_comp_video)
        
    # 3-2. 압축된 segment_comp_videos를 ZIP 파일로 압축
    with zipfile.ZipFile(segment_zip_path, 'w') as zipf:
        for video in segment_comp_videos:
            zipf.write(video, os.path.basename(video))  # ZIP 파일에 파일을 추가
    print(f"ZIP file created at: {segment_zip_path}")

    for file in segment_comp_videos:
        os.remove(file)

    # 4-1. 가장 긴 세그먼트에 해당하는 비디오 생성
    highlight_video_paths = []
    for i, segment in enumerate(highlight_segments):
        _highlight_temp_video_path = highlight_temp_video_path.replace(".mp4", f"_{i+1}.mp4")
        _highlight_video_path = highlight_video_path.replace(".mp4", f"_{i+1}.mp4")
        
        # 각 세그먼트를 하이라이트 비디오로 생성
        create_highlight_video(video_file_path, segment, _highlight_temp_video_path)
        compress_video_with_ffmpeg(_highlight_temp_video_path, _highlight_video_path)
        
        # 생성된 하이라이트 비디오를 리스트에 추가
        highlight_video_paths.append(_highlight_video_path)
        
        # 원본 하이라이트 비디오 삭제 (압축된 비디오만 유지)
        os.remove(_highlight_temp_video_path)

    # 4-2. 압축된 highlight_comp_videos를 ZIP 파일로 압축
    with zipfile.ZipFile(highlight_zip_path, 'w') as zipf:
        for video in highlight_video_paths:
            zipf.write(video, os.path.basename(video))  # ZIP 파일에 파일을 추가
    print(f"Highlight ZIP file created at: {highlight_zip_path}")

    for file in highlight_video_paths:
        os.remove(file)

    return highlight_zip_path, segment_zip_path


# 한 청크를 처리하고 예측 결과를 반환하는 함수 (target_fps로 예측)
def process_chunk_for_prediction(cap, model, val_transform, chunk_size, frame_index, target_fps, resize_dim=(1280, 720)):
    frames = []
    for _ in range(chunk_size):
        cap.set(cv2.CAP_PROP_POS_MSEC, frame_index * (1000 / target_fps))
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 리사이즈
        resized_frame = cv2.resize(frame, resize_dim)

        frames.append(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
        frame_index += 1

    if len(frames) < chunk_size:
        return None, frame_index

    # NumPy 배열로 변환한 후 텐서로 변환
    video_array = np.array(frames, dtype=np.float32)
    video_tensor = torch.tensor(video_array)

    # 예측 수행
    predicted_class, predicted_class_idx, probs = predict_video_class(video_tensor, model, val_transform)
    playing_prob = probs[1]  # playing 클래스 확률
    not_playing_prob = probs[0]  # not-playing 클래스 확률

    return (predicted_class, playing_prob, not_playing_prob), frame_index


from torch.cuda.amp import autocast

# 모델 인퍼런스 함수 정의
def predict_video_class(video, model, val_transform):
    # 레이블 맵핑
    id2label = {0: "not-playing", 1: "playing"}

    # (T, H, W, C) -> (C, T, H, W)로 변환
    video = video.permute(3, 0, 1, 2)
    # 전처리를 적용하여 동일한 크기로 변환
    video = val_transform({"video": video})["video"]
    # (C, T, H, W) -> (1, T, C, H, W) 배치 차원 추가
    inputs = video.unsqueeze(0).permute(0, 2, 1, 3, 4).to(config.CUDA_DEVICE)
    
    # 모델 예측 수행
    with torch.no_grad():
        with autocast():  # Mixed precision for speedup
            outputs = model(inputs)
    
    # 예측된 확률과 클래스
    probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
    predicted_class_idx = outputs.logits.argmax(-1).item()
    predicted_class = id2label[predicted_class_idx]
    
    return predicted_class, predicted_class_idx, probs


# 예측 결과를 이미지에 텍스트로 추가
def add_prediction_text_to_frame(frame, playing_prob, not_playing_prob, time_start, time_end):
    
    frame_copy = frame.copy()
    
    # 텍스트를 이미지에 추가
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color_playing = (0, 255, 0)  # playing일 때 녹색
    color_not_playing = (0, 0, 255)  # not-playing일 때 빨강색
    thickness = 2
    position_playing = (10, 30)  # 좌상단 위치
    position_not_playing = (10, 60)  # 그 아래 위치
    position_time = (10, 90)  # 그 아래에 시간 텍스트 위치 추가

    # playing 확률 텍스트
    playing_text = f"Playing: {playing_prob:.2f}"
    # not-playing 확률 텍스트
    not_playing_text = f"Not Playing: {not_playing_prob:.2f}"
    # 청크의 시간 범위 텍스트 추가 (소수점 한 자리까지)
    time_text = f"Time: {time_start:.1f}~{time_end:.1f}s"

    # 텍스트를 이미지에 추가
    cv2.putText(frame_copy, playing_text, position_playing, font, font_scale, color_playing, thickness, cv2.LINE_AA)
    cv2.putText(frame_copy, not_playing_text, position_not_playing, font, font_scale, color_not_playing, thickness, cv2.LINE_AA)
    cv2.putText(frame_copy, time_text, position_time, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return frame_copy

def initialize_model():
    device = config.CUDA_DEVICE
    model_checkpoint = config.MODEL_CHECKPOINT
    model = VideoMAEForVideoClassification.from_pretrained(model_checkpoint).to(device)
    
    image_processor = VideoMAEImageProcessor.from_pretrained(model_checkpoint)
    mean = image_processor.image_mean
    std = image_processor.image_std
    height = width = image_processor.size.get("shortest_edge")
    resize_to = (height, width)

    val_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(model.config.num_frames),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        Resize(resize_to),
                    ]
                ),
            ),
        ]
    )
    return model, val_transform


def process_video(self, video_path, process_options, username):
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    print(f"process_video 실행: video_path: {video_path}, video_filename:{video_filename}, process_options:{process_options}")
    
    model, val_transform = initialize_model()
    
    # 1. Target FPS로 비디오에서 예측 결과 생성
    predictions, original_fps = process_video_for_predictions(video_path, model, val_transform)

    # 2. process_options에 따라서 Make 여부 판단
    make_full = 'Full' in process_options
    make_playing = 'Playing' in process_options
    make_highlight = 'Highlight' in process_options

    print(f"make_full:{make_full}, make_playing:{make_playing}, make_highlight:{make_highlight}")

    base_video_folder = os.path.join(config.VIDEO_DATA_FOLDER, username, "processed")

    # 폴더가 없으면 생성
    if not os.path.exists(base_video_folder):
        os.makedirs(base_video_folder, exist_ok=True)

    video_filename_uuid = video_filename + str(uuid.uuid4())[:8]
    target_folder_path = os.path.join(base_video_folder, video_filename_uuid)
    os.makedirs(target_folder_path, exist_ok=True)

    # 3. 비디오 출력 경로 설정 및 생성
    if make_full:        
        full_temp_video_path = os.path.join(target_folder_path, f"temp_full.mp4")
        full_video_path = full_temp_video_path.replace("temp_full", "full")
        print(f"Let's make full video, path:{full_video_path}")
        extract_type = "full"
        full_video_path = extract_video_using_prediction(video_path, full_temp_video_path, full_video_path, predictions, original_fps, extract_type)
    else:
        full_video_path = None
    
    if make_playing:        
        playing_temp_video_path = os.path.join(target_folder_path, f"temp_playing.mp4")
        playing_video_path = playing_temp_video_path.replace("temp_playing", "playing")
        print(f"Let's make playing video, path:{playing_video_path}")
        extract_type = "playing"
        playing_video_path = extract_video_using_prediction(video_path, playing_temp_video_path, playing_video_path, predictions, original_fps, extract_type)
    else:
        playing_video_path = None

    if make_highlight:
        highlight_temp_video_path = os.path.join(target_folder_path, f"temp_highlight.mp4")
        highlight_video_path = highlight_temp_video_path.replace("temp_highlight", "highlight")
        segment_zip_path = os.path.join(target_folder_path, f"segment.zip")
        highlight_zip_path = os.path.join(target_folder_path, f"highlight.zip")
        
        print(f"Let's make highlight video, path:{highlight_video_path}")
        print(f"Let's make segment videos, path:{segment_zip_path}")
        highlight_zip_path, segment_zip_path = extract_videos_using_prediction(predictions, video_path, highlight_temp_video_path, highlight_video_path, target_folder_path, segment_zip_path, highlight_zip_path)
    else:
        highlight_zip_path = None
        segment_zip_path = None
        
    print(f"process_video finished")
    print(full_video_path, playing_video_path, highlight_zip_path, segment_zip_path)

    return full_video_path, playing_video_path, highlight_zip_path, segment_zip_path
    
