# Nvidia PyTorch 이미지 기반
FROM nvcr.io/nvidia/pytorch:24.03-py3

# 작업 디렉토리 설정
WORKDIR /base

# Miniconda 설치
RUN apt-get update && apt-get install -y wget bzip2 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Miniconda 환경변수 설정
ENV PATH="/opt/conda/bin:$PATH"

# conda 환경 생성 및 Python 패키지 설치
COPY ./requirements.txt /base/
RUN conda install python=3.10 && \
    conda create --name myenv python=3.10 && \
    conda init bash && \
    echo "conda activate myenv" >> ~/.bashrc && \
    /opt/conda/envs/myenv/bin/pip install --upgrade pip

RUN /opt/conda/envs/myenv/bin/pip install -r /base/requirements.txt

# FFmpeg 및 기타 필수 라이브러리 설치
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

# NVIDIA GPG 키 추가, 리포지토리 추가 및 설치
RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && \
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - && \
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list && \
    apt-get update && apt-get install -y nvidia-container-toolkit

# 로그 디렉토리 생성
RUN mkdir -p logs/celery logs/gradio logs/uvicorn app/src

# 애플리케이션 파일 복사
COPY ./run.sh /base/
COPY ./model /base/model

# 포트 설정
EXPOSE 41090

# 쉘 스크립트를 실행하도록 설정 (Celery와 Uvicorn을 백그라운드에서 실행)
ENTRYPOINT ["bash", "run.sh"]
#ENTRYPOINT ["sleep", "infinity"]