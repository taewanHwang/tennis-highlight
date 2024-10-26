read -p "정말로 실행하시겠습니까? (y/n): " answer

# 입력한 답이 'y'인 경우에만 명령어를 실행합니다.
if [ "$answer" == "y" ]; then
    docker-compose down
    docker build -t practice2_image .
    docker-compose up --build -d
else
    echo "실행이 취소되었습니다."
fi
