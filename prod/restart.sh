#!/bin/bash
###########################
#./restart.sh : practice 컨테이너만 재시작(코드만 수정됐을 때)
#./restart.sh -a : 레디스, 네트워크 포함 전체 재시작(도커빌드 없이)
#./restart.sh -b : Dockerfile이 변경됐을때 이미지 빌드
#./restart.sh -s : 전체 종료
###########################
# 기본값으로는 practice3_container만 재시작
RESTART_MODE="practice2_container"
BUILD_MODE=false
STOP_MODE=false

# 옵션 파싱
while getopts "abs" opt; do
  case $opt in
    a)
      RESTART_MODE="all"  # 전체 재시작 모드
      ;;
    b)
      BUILD_MODE=true  # 빌드 모드
      ;;
    s)
      STOP_MODE=true  # 서비스 종료 모드
      ;;
    \?)
      echo "잘못된 옵션: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

if [ "$STOP_MODE" == true ]; then
    # 전체 서비스 종료
    echo "서비스를 종료합니다."
    docker-compose down
    exit 0
fi

# 실행 전 사용자에게 확인
read -p "정말로 실행하시겠습니까? (y/n): " answer

if [ "$answer" == "y" ]; then
    if [ "$BUILD_MODE" == true ]; then
        # 빌드 모드 (캐시 없이 빌드 후 전체 재시작)
        docker-compose down
        docker build -t practice2_image .
        docker-compose up --build -d
        docker image prune -f
    else
        if [ "$RESTART_MODE" == "all" ]; then
            # 전체 재시작 (빌드 없이)
            docker-compose down
            docker-compose up -d
        else
            # practice3_container만 재시작 (다른 컨테이너는 영향 없음)
            docker-compose stop practice2_container
            docker-compose up -d practice2_container
        fi
    fi
else
    echo "실행이 취소되었습니다."
fi


