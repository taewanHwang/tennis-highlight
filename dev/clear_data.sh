#!/bin/bash

# 확인 메시지 출력
read -p "Are you sure you want to delete all files and folders under video_data? (y/n): " confirm

# 입력 값이 y일 때만 삭제 수행
if [ "$confirm" = "y" ]; then
    sudo rm -rf /home/disk5/taewan/study/practice2/demo3/dev/data/video_data/*
    echo "Files and folders deleted."
else
    echo "Operation canceled."
fi
