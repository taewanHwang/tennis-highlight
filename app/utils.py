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
import app.config as config
import cv2
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import time
import zipfile

def format_filename(title, max_length=50):
    # 파일명에 적합하지 않은 문자 제거 (파일명으로 사용할 수 없는 문자)
    title = re.sub(r'[\\/*?:"<>|]', "", title)
    # 길이가 50자를 넘으면 자르기
    return title[:max_length]

def download_youtube_video(youtube_link, start_time, end_time):
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 유튜브 제목 가져오기
    with yt_dlp.YoutubeDL() as ydl:
        info_dict = ydl.extract_info(youtube_link, download=False)
        video_title = info_dict.get('title', 'Unknown Title')

    # 제목 포맷팅 (50자 제한 및 특수문자 제거)
    formatted_title = format_filename(video_title)

    # 파일명 생성 (유튜브 제목 + 현재 시간)
    output_filename = f"{formatted_title}_{current_time}.mp4"
    output_path = os.path.join(config.TEMP_FOLDER, output_filename)

    ydl_opts = {
        'outtmpl': output_path,
        'format': 'bestvideo+bestaudio/best',
        'postprocessor_args': ['-ss', start_time, '-to', end_time],
        'merge_output_format': 'mp4',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_link])

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
        predictions = postprocess_predictions(predictions, config)

    print(f"predictions:{predictions}")
    print(f"original_fps:{original_fps}")

    # 예측 결과를 반환
    return predictions, original_fps

def postprocess_predictions(predictions, config, n=5):
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

def extract_playing_segments_from_video(predictions, video_file_path, highlight_video_path, highlight_comp_video_path, folder_path, segment_zip_path, make_highlights, make_segments):
    chunk_duration = predictions[0][0]  # 각 청크의 지속 시간

    # 1. 연속적인 playing 구간 찾기
    playing_segments = find_playing_segments(predictions, chunk_duration)
    print(f"Playing Segments: {playing_segments}")

    # 2. 가장 긴 세그먼트 찾기
    longest_segments = find_longest_segments(playing_segments, n=5)
    print(f"Longest Segment: {longest_segments}")

    # 비디오1: 연속적인 세그먼트에 해당하는 비디오 생성
    if make_segments:
        segment_videos = create_segment_videos(video_file_path, playing_segments, folder_path)
        print(f"segment_videos: {segment_videos}")
        segment_comp_videos = []  # 압축된 비디오 파일들을 저장할 리스트
        for segment_video in segment_videos:
            segment_comp_video = segment_video.replace("temp_", "")
            compress_video_with_ffmpeg(segment_video, segment_comp_video) 
            segment_comp_videos.append(segment_comp_video)
        # 압축된 segment_comp_videos를 ZIP 파일로 압축
        with zipfile.ZipFile(segment_zip_path, 'w') as zipf:
            for video in segment_comp_videos:
                zipf.write(video, os.path.basename(video))  # ZIP 파일에 파일을 추가

        print(f"ZIP file created at: {segment_zip_path}")

        for segment_video in segment_videos:
            os.remove(segment_video)

    # 비디오2: 가장 긴 세그먼트에 해당하는 비디오 생성
    highlight_videos = []
    if make_highlights:
        for i, segment in enumerate(longest_segments):
            temp_highlight_video_path = highlight_video_path.replace(".mp4", f"_{i+1}.mp4")
            temp_highlight_comp_video_path = highlight_comp_video_path.replace(".mp4", f"_{i+1}.mp4")
            
            # 각 세그먼트를 하이라이트 비디오로 생성
            create_highlight_video(video_file_path, segment, temp_highlight_video_path)
            compress_video_with_ffmpeg(temp_highlight_video_path, temp_highlight_comp_video_path)
            
            # 생성된 하이라이트 비디오를 리스트에 추가
            highlight_videos.append(temp_highlight_comp_video_path)
            
            # 원본 하이라이트 비디오 삭제 (압축된 비디오만 유지)
            os.remove(temp_highlight_video_path)

        print(f"Highlight videos created: {highlight_videos}")
    return highlight_videos
    

def extract_playing_from_video(video_file_path, full_video_path_1, full_comp_video_path_1, playing_video_path_2, playing_comp_video_path_2, predictions, original_fps, threshold=config.PLAY_THRESHOLD, make_full=False, make_playing=True):    
    cap = cv2.VideoCapture(video_file_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 비디오1 (전체 플레이 영상, 텍스트 포함)
    out_all = cv2.VideoWriter(full_video_path_1, fourcc, original_fps, (width, height))
    if not out_all.isOpened():
        raise IOError(f"Error: Failed to open video writer for {full_video_path_1}. Ensure that codec is supported.")

    # 비디오2 (playing_prob > threshold 인 구간만 저장)
    playing_out = cv2.VideoWriter(playing_video_path_2, fourcc, original_fps, (width, height))
    if not playing_out.isOpened():
        raise IOError(f"Error: Failed to open video writer for {playing_video_path_2}. Ensure that the codec is supported.")

    previous_end_frame = 0
    for idx, (time_index, pred_label, playing_prob, not_playing_prob) in enumerate(predictions):
        start_frame = previous_end_frame
        end_frame = int(time_index * original_fps)
        print(f"start_frame:{start_frame}, end_frame:{end_frame}, time_index:{time_index}")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 프레임 읽고 처리
        frame_start_time = time.time()
        for frame_index in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            frame_with_text = add_prediction_text_to_frame(frame, playing_prob, not_playing_prob, start_frame / original_fps, end_frame / original_fps)
            if make_full:
                out_all.write(frame_with_text)  # 비디오1에 전체 프레임 저장
            
            if (playing_prob > threshold) and (make_playing == True):
                playing_out.write(frame)  # 비디오2에 해당 프레임 저장
        
        # 다음 구간을 위한 end_frame 업데이트
        previous_end_frame = end_frame

    cap.release()
    out_all.release()
    playing_out.release()

    print(f"Full video with text saved at: {full_video_path_1}")
    print(f"Playing segments video saved at: {playing_video_path_2}")

    # 압축 작업 시간 측정
    compress_video_with_ffmpeg(full_video_path_1, full_comp_video_path_1)  # 전체 비디오 압축
    compress_video_with_ffmpeg(playing_video_path_2, playing_comp_video_path_2)  # 플레이 비디오 압축
    os.remove(full_video_path_1)
    os.remove(playing_video_path_2)

    print(f"Compressed and saved at: {full_comp_video_path_1}, {playing_comp_video_path_2}")



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


# 차트를 그리는 함수 추가
def plot_probabilities(predictions, video_filename, output_chart_dir="/base/app/plot"):
    print(f"plot_probabilities")
    times = [pred[0] for pred in predictions]  # 예측된 시간 리스트
    playing_probs = [pred[2] for pred in predictions]  # Playing 클래스에 대한 확률 리스트
    moving_avg = np.convolve(playing_probs, np.ones(3)/3, mode='same')  # same 모드로 계산해서 길이 맞춤

    plt.figure(figsize=(10, 6))
    
    # Playing Probability에 대한 라인 플롯 (검정색 마커)
    plt.plot(times, playing_probs, marker='o', color='black', label='Playing Probability')
    
    # Moving Average 라인 추가 (파란색)
    plt.plot(times, moving_avg, marker='o', color='blue', label='Moving Average (3)')

    # y=0.5 빨간색 대시 라인 추가
    plt.axhline(y=0.5, color='red', linestyle='--', label='Threshold (0.5)')

    # 라벨 추가
    plt.xlabel('Time (s)')
    plt.ylabel('Playing Probability')
    plt.title('Playing Probability over Time')
    plt.legend()
    
    output_chart_path = os.path.join(output_chart_dir, f"{video_filename}.png")

    # 차트를 파일로 저장
    plt.savefig(output_chart_path)

    print(f"Chart saved at: {output_chart_path}")

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

import redis

# Redis 설정
r = redis.Redis(host='redis_container', port=6379, db=0)

def set_stop_flag(task_id):
    """특정 task_id에 대해 중지 플래그 설정"""
    r.set(f"stop_task_{task_id}", "true")

def clear_stop_flag(task_id):
    """특정 task_id에 대해 중지 플래그 해제"""
    r.delete(f"stop_task_{task_id}")

def is_task_stopped(task_id):
    """특정 task_id에 대해 중지 플래그 확인"""
    return r.get(f"stop_task_{task_id}") == b"true"


def process_video(self, video_path, make_full=False, make_playing=False, make_highlights=True, make_segments=False):
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    print(f"process_video 실행: video_path: {video_path}, video_filename:{video_filename}")

    short_uuid = video_filename + str(uuid.uuid4())[:8]
    folder_path = os.path.join(config.PROCESSED_VIDEOS_FOLDER, short_uuid)
    os.makedirs(folder_path, exist_ok=True)

    # 2. 비디오 출력 경로 설정
    full_temp_video_path = os.path.join(folder_path, f"temp_full.mp4")
    playing_temp_video_path = os.path.join(folder_path, f"temp_playing.mp4")
    highlight_temp_video_path = os.path.join(folder_path, f"temp_highlight.mp4")
    segment_zip_path = os.path.join(folder_path, f"segment.zip")

    full_video_path = full_temp_video_path.replace("temp_full", "full")
    playing_video_path = playing_temp_video_path.replace("temp_playing", "playing")
    highlight_video_path = highlight_temp_video_path.replace("temp_highlight", "highlight")

    print(f"full video path:{full_video_path}")
    print(f"playing video path:{playing_video_path}")
    print(f"highlight video path:{highlight_video_path}")
    print(f"segment zip path:{segment_zip_path}")

    model, val_transform = initialize_model()
    
    # 3. Target FPS로 비디오에서 예측 결과 생성
    print(f'task_id:{self.request.id}, is_stoppped:{is_task_stopped(self.request.id)}')
    if is_task_stopped(self.request.id):
        self.update_state(state='REVOKED', meta={"info": "Task was stopped by user."})
        return {"message": "Task was stopped by user"}
    predictions, original_fps = process_video_for_predictions(video_path, model, val_transform)


    # 4. 예측 결과를 바탕으로 full, playing 추출
    print(f'task_id:{self.request.id}, is_stoppped:{is_task_stopped(self.request.id)}')
    if is_task_stopped(self.request.id):
        self.update_state(state='REVOKED', meta={"info": "Task was stopped by user."})
        return {"message": "Task was stopped by user"}
    extract_playing_from_video(video_path, full_temp_video_path, full_video_path, 
                                        playing_temp_video_path, playing_video_path,
                                        predictions, original_fps, make_full, make_playing)
    
    # 5. 예측 결과를 바탕으로 highlight, segment zip 추출
    print(f'task_id:{self.request.id}, is_stoppped:{is_task_stopped(self.request.id)}')
    if is_task_stopped(self.request.id):
        self.update_state(state='REVOKED', meta={"info": "Task was stopped by user."})
        return {"message": "Task was stopped by user"}
    highlight_video_paths = extract_playing_segments_from_video(predictions, video_path, highlight_temp_video_path, highlight_video_path, folder_path, segment_zip_path, make_highlights, make_segments)
    print(f"result highlight video path:{highlight_video_paths}")

    print(f"비디오 처리 완료: full_video_path={full_video_path}, playing_video_path={playing_video_path}, highlight_video_path={highlight_video_path}")

    return full_video_path, playing_video_path, highlight_video_paths, segment_zip_path


# 파일에서 사용자 정보 불러오기
def load_users(file_path=config.AUTH_FILEPATH):
    # print(f'load auth file:{file_path}')
    users = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            # 줄에서 사용자 이름과 비밀번호를 쉼표로 분리
            username, password = line.strip().split(",")
            users.append((username, password))
    return users

