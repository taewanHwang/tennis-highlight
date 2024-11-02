import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
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
import subprocess
import zipfile
from datetime import datetime, timezone, timedelta
import re
from urllib.parse import urlparse
from pydub import AudioSegment
import ffmpeg

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

class loggerOutputs:
    def error(msg):
        print(f"[Error] {msg}",flush=True)
    def warning(msg):
        print(f"[Warning] {msg}",flush=True)
    def debug(msg):
        pass

import os
from datetime import datetime
import yt_dlp
import subprocess

def download_youtube_video(youtube_link, start_time, end_time, username):
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 유튜브 제목 가져오기
    with yt_dlp.YoutubeDL() as ydl:
        info_dict = ydl.extract_info(youtube_link, download=False)
        video_title = info_dict.get('title', 'Unknown Title')

    # 제목 포맷팅 (50자 제한 및 특수문자 제거)
    formatted_title = format_filename(video_title)

    # username 폴더 경로 생성
    video_temp_folder = os.path.join(config.VIDEO_DATA_FOLDER, username, "temp")

    # 폴더가 없으면 생성
    if not os.path.exists(video_temp_folder):
        os.makedirs(video_temp_folder, exist_ok=True)

    # 원본 다운로드 경로 생성
    raw_output_filename = f"{formatted_title}_{current_time}_raw.mp4"
    raw_output_path = os.path.join(video_temp_folder, raw_output_filename)

    # 최종 자른 파일 경로 생성
    final_output_filename = f"{formatted_title}_{current_time}.mp4"
    final_output_path = os.path.join(video_temp_folder, final_output_filename)

    # yt-dlp 옵션 설정 (비디오와 오디오 스트림 다운로드)
    ydl_opts = {
        'outtmpl': raw_output_path,
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'quiet': True,
        'no_warnings': True,
        'verbose': False,
        "logger": loggerOutputs,
    }

    # 유튜브 비디오 다운로드
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_link])


    ffmpeg_command = [
        'ffmpeg',
        '-ss', start_time,  # 입력 파일에 직접 -ss 적용
        '-to', end_time,    # 입력 파일에 직접 -to 적용
        '-i', raw_output_path,
        '-c', 'copy',
        final_output_path,
        '-loglevel', 'error'  # 에러만 출력하여 콘솔 로그 간소화
    ]

    # FFmpeg 명령 실행
    subprocess.run(ffmpeg_command, check=True)

    # 원본 다운로드 파일 삭제
    os.remove(raw_output_path)

    # 문자열 시간을 초 단위로 변환하는 함수
    def time_str_to_seconds(time_str):
        h, m, s = map(float, time_str.split(":"))
        return int(timedelta(hours=h, minutes=m, seconds=s).total_seconds())

    # 예상 지속 시간 (초)
    expected_duration = time_str_to_seconds(end_time) - time_str_to_seconds(start_time)

    # 비디오 스트림의 길이 확인
    ffprobe_video_command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',  # 첫 번째 비디오 스트림 선택
        '-show_entries', 'stream=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        final_output_path
    ]
    video_duration = float(subprocess.check_output(ffprobe_video_command).strip())

    # 오디오 스트림의 길이 확인
    ffprobe_audio_command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',  # 첫 번째 오디오 스트림 선택
        '-show_entries', 'stream=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        final_output_path
    ]
    audio_duration = float(subprocess.check_output(ffprobe_audio_command).strip())

    # 비디오와 오디오 스트림 길이 비교
    success = True
    error_margin = abs(expected_duration) * 0.002 # 전체 길이의 0.2%
    if abs(video_duration - expected_duration) > error_margin: 
        print(f"경고: 비디오 길이({video_duration}초)가 예상 길이({expected_duration}초)와 일치하지 않습니다. 에러 허용은 {error_margin}초까지 가능합니다.")
        success = False

    if abs(audio_duration - expected_duration) > error_margin: 
        print(f"경고: 오디오 길이({audio_duration}초)가 예상 길이({expected_duration}초)와 일치하지 않습니다. 에러 허용은 {error_margin}초까지 가능합니다.")
        success = False

    return final_output_path if success else None

def process_video_for_predictions(video_file_path, model, val_transform, chunk_size=16, target_fps=6, postprocess=True):
    cap = cv2.VideoCapture(video_file_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)  # 원본 fps 가져오기
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 비디오의 총 프레임 수 가져오기

    print(f"video_file_path: {video_file_path}, Total number of frames: {total_frames}, original_fps: {original_fps}")

    frame_index = 0
    predictions = []

    while cap.isOpened():
        # target_fps로 청크를 처리하고 예측 결과 받기
        prediction, frame_index = process_chunk_for_prediction(cap, model, val_transform, chunk_size, frame_index, target_fps)
        cur_time = frame_index/target_fps
        
        if prediction is None:
            break  # 프레임이 충분하지 않으면 종료

        predicted_class, playing_prob, not_playing_prob = prediction
        predictions.append((frame_index / target_fps, predicted_class, playing_prob, not_playing_prob))

    cap.release()

    # 후처리 적용
    if postprocess:
        predictions = postprocess_predictions(predictions, config.POST_CONS_CHUNK_SIZE)

    print(f"predictions:{predictions}")

    # 예측 결과를 반환
    return predictions, original_fps

def postprocess_predictions(predictions, n):
    """
    연속 n개의 청크가 playing이고, 현재 청크가 not-playing인 경우 후처리하여 playing으로 변경.
    또한, not-playing -> playing -> not-playing 형태의 1순간 playing 구간을 not-playing으로 변환.
    후처리가 적용된 이후, 새로운 playing 구간이 발생하면 후처리 상태를 초기화함.
    """
    play_streak = []  # 최근 n개의 청크 저장용
    processed_predictions = []
    postprocess_applied = False  # 후처리가 한 번 적용되었는지 확인하는 플래그

    # 첫 번째 후처리: 연속 n개의 playing 후에 not-playing을 playing으로 변경
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

    # 두 번째 후처리: not-playing -> playing -> not-playing 구간을 not-playing으로 변경
    final_predictions = []
    i = 0
    while i < len(processed_predictions):
        # 현재 청크가 'playing'이고 이전과 다음 청크가 모두 'not-playing'이면 후처리
        if (
            i > 0 and i < len(processed_predictions) - 1
            and processed_predictions[i - 1][1] == 'not-playing'
            and processed_predictions[i][1] == 'playing'
            and processed_predictions[i + 1][1] == 'not-playing'
        ):
            # 현재 청크를 'not-playing'으로 변환
            time_stamp, _, playing_prob, not_playing_prob = processed_predictions[i]
            final_predictions.append((time_stamp, 'not-playing', 0.1, 0.9))
            i += 1  # 중간 청크 건너뜀
        else:
            final_predictions.append(processed_predictions[i])
            i += 1

    return final_predictions

def compress_video_with_ffmpeg(input_path, output_path):
    command = [
        'ffmpeg',
        '-i', input_path,           # 입력 파일 경로
        '-vcodec', 'libx264',       # H.264 코덱 사용
        '-crf', '23',               # 품질 설정 (23이 기본, 낮을수록 품질이 높아지고 용량이 커짐)
        '-preset', 'medium',        # 속도와 압축의 균형 (faster, medium, slower 등 선택 가능)
        '-y',                       # 기존 파일 덮어쓰기
        output_path                 # 출력 파일 경로
    ]


    try:
        # ffmpeg 명령 실행 및 stderr 캡처
        result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, check=True)
        print(f"Video successfully compressed and saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print("Error occurred while compressing video:", e.stderr.decode())
        raise

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
    # 비디오 파일 읽기 및 기본 설정
    cap = cv2.VideoCapture(video_file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 원본 프레임 형식 유지

    # 임시 비디오 및 오디오 경로 생성
    temp_video_path = output_path.replace(".mp4", "_temp.mp4")
    temp_audio_path = output_path.replace(".mp4", "_temp_audio.wav")
    temp_full_audio_path = output_path.replace(".mp4", "_temp_full_audio.wav")

    # 전체 오디오 추출
    ffmpeg.input(video_file_path).output(temp_full_audio_path, format='wav').run(quiet=True)
    original_audio = AudioSegment.from_file(temp_full_audio_path)

    # 비디오 라이터 초기화
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    # 프레임 구간 계산
    start_time, end_time = longest_segment
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # 비디오 프레임 추출
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    out.release()
    cap.release()

    # 오디오 구간 추출 및 저장
    start_ms = start_time * 1000  # ms 단위
    end_ms = end_time * 1000
    audio_segment = original_audio[start_ms:end_ms]
    audio_segment.export(temp_audio_path, format="wav")

    # 비디오와 오디오 병합
    video_stream = ffmpeg.input(temp_video_path)
    audio_stream = ffmpeg.input(temp_audio_path)
    ffmpeg.concat(video_stream, audio_stream, v=1, a=1).output(output_path).run()

    # 임시 파일 삭제
    os.remove(temp_video_path)
    os.remove(temp_audio_path)
    os.remove(temp_full_audio_path)

    print(f"Created highlight video with audio: {output_path}")

def create_segment_videos(video_file_path, segments, base_dir):
    # 비디오 파일 읽기 및 기본 설정
    cap = cv2.VideoCapture(video_file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 원본 프레임 형식 유지


    # 전체 오디오 로드
    temp_audio_file = os.path.join(base_dir, "temp_full_audio.wav")
    ffmpeg.input(video_file_path).output(temp_audio_file, format='wav').run(quiet=True)
    original_audio = AudioSegment.from_file(temp_audio_file)

    segment_videos = []

    for idx, (start_time, end_time) in enumerate(segments):
        # 비디오 및 오디오 파일 경로 생성
        output_file = os.path.join(base_dir, f'segment_temp{idx + 1}.mp4')
        temp_video_path = os.path.join(base_dir, f'segment_video_temp{idx + 1}.mp4')
        temp_audio_path = os.path.join(base_dir, f'segment_audio_temp{idx + 1}.wav')

        # 비디오 라이터 초기화
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

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
        
        # 오디오 구간 추출 및 저장
        start_ms = start_time * 1000  # ms 단위
        end_ms = end_time * 1000
        audio_segment = original_audio[start_ms:end_ms]
        audio_segment.export(temp_audio_path, format="wav")

        # 비디오와 오디오 병합
        video_stream = ffmpeg.input(temp_video_path)
        audio_stream = ffmpeg.input(temp_audio_path)
        ffmpeg.concat(video_stream, audio_stream, v=1, a=1).output(output_file).run()

        segment_videos.append(output_file)

        # 임시 파일 삭제
        os.remove(temp_video_path)
        os.remove(temp_audio_path)

    # 전체 임시 오디오 파일 삭제
    os.remove(temp_audio_file)
    cap.release()

    return segment_videos

def extract_video_using_prediction(video_path, temp_video_path, output_video_path, predictions, original_fps, extract_type, play_threshold=config.PLAY_THRESHOLD):
    # 비디오 설정 초기화
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cvwriter = cv2.VideoWriter(temp_video_path, fourcc, original_fps, (width, height))

    # video_path 파일명에서 확장자를 .wav로 변경하고, 끝에 _temp_audio 추가
    video_dir = os.path.dirname(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    temp_audio_file = os.path.join(video_dir, f"{base_name}_temp_audio.wav")

    # 기존에 임시 파일이 있으면 삭제
    if os.path.exists(temp_audio_file):
        os.remove(temp_audio_file)

    # 임시 오디오 파일 추출 및 로드
    ffmpeg.input(video_path).output(temp_audio_file, format='wav').run(quiet=True)
    original_audio = AudioSegment.from_file(temp_audio_file)

    # 병합된 오디오 클립을 저장할 객체 초기화
    merged_audio = AudioSegment.empty()

    previous_end_frame = 0
    for idx, (time_index, pred_label, playing_prob, not_playing_prob) in enumerate(predictions):
        start_frame = previous_end_frame
        end_frame = int(time_index * original_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 비디오 프레임 처리와 오디오 병합 조건 확인
        should_write_audio = False

        for frame_index in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            # 조건에 맞는 프레임과 오디오 병합 결정
            if extract_type == "full":
                frame_with_text = add_prediction_text_to_frame(frame, playing_prob, not_playing_prob, start_frame / original_fps, end_frame / original_fps)
                cvwriter.write(frame_with_text)
                should_write_audio = True
            elif extract_type == "playing" and playing_prob > play_threshold:
                cvwriter.write(frame)
                should_write_audio = True

        # 오디오 구간 추출 및 병합
        if should_write_audio:
            start_ms = (start_frame / original_fps) * 1000
            end_ms = (end_frame / original_fps) * 1000
            audio_segment = original_audio[start_ms:end_ms]
            merged_audio += audio_segment

        previous_end_frame = end_frame

    cap.release()
    cvwriter.release()

    # 병합된 오디오를 임시 파일로 저장 (임시 파일 이름 동일 방식으로 생성)
    temp_merged_audio_path = os.path.join(video_dir, f"{base_name}_temp_merged_audio.wav")
    if os.path.exists(temp_merged_audio_path):
        os.remove(temp_merged_audio_path)
    merged_audio.export(temp_merged_audio_path, format="wav")

    # 최종 비디오와 오디오 병합
    try:
        video_stream = ffmpeg.input(temp_video_path)
        audio_stream = ffmpeg.input(temp_merged_audio_path)
        ffmpeg.output(video_stream, audio_stream, output_video_path, vcodec='copy', acodec='aac').run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        print("ffmpeg error:", e.stderr.decode())
        raise e
    finally:
        # 임시 파일 삭제
        os.remove(temp_video_path)
        os.remove(temp_audio_file)
        os.remove(temp_merged_audio_path)

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
        segment_comp_video = segment_video.replace("_temp", "")
        print(f"segment_video:{segment_video}, segment_comp_video:{segment_comp_video}")
        compress_video_with_ffmpeg(segment_video, segment_comp_video) 
        segment_comp_videos.append(segment_comp_video)
        
    # 3-2. 압축된 segment_comp_videos를 ZIP 파일로 압축
    with zipfile.ZipFile(segment_zip_path, 'w') as zipf:
        for video in segment_comp_videos:
            zipf.write(video, os.path.basename(video))  # ZIP 파일에 파일을 추가
    print(f"ZIP file created at: {segment_zip_path}")

    for file in segment_videos:
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
        set_time = frame_index * (1000 / target_fps)
        cap.set(cv2.CAP_PROP_POS_MSEC, set_time)
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

    set_time_end = frame_index * (1000 / target_fps)
    set_time_start = (frame_index-chunk_size) * (1000 / target_fps)
        
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
        
    print(f"****** process_video finished ******")
    print(f"Full Video Path: {full_video_path}, "
        f"Playing Video Path: {playing_video_path}, "
        f"Highlight Zip Path: {highlight_zip_path}, "
        f"Segment Zip Path: {segment_zip_path}")
    print(f"************************************")

    return full_video_path, playing_video_path, highlight_zip_path, segment_zip_path
    
