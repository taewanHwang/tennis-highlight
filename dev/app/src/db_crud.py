from sqlalchemy.orm import Session, joinedload
from db_models import User, VideoTask, VideoTaskType, VideoResult
from datetime import datetime, timezone
from db import SessionLocal  
import pandas as pd
from fastapi import HTTPException
from utils import convert_to_kst, format_datetime_short
import config

def load_users_from_db(db: Session):
    # MySQL에서 모든 사용자 정보 조회
    users = db.query(User).all()
    user_list = []
    
    # 각 사용자에 대해 (username, password_hash) 형식으로 변환
    for user in users:
        user_list.append((user.username, user.password_hash))
    
    return user_list

def authenticate_user(username: str, password: str):
    # password_hash = hashlib.sha256(password.encode()).hexdigest()
    password_hash = password
    with SessionLocal() as db:
        user = db.query(User).filter(User.username == username, User.password_hash == password_hash).first()
    return user is not None

def cancel_video_task(req_id: str):
    with SessionLocal() as db:  # 세션을 함수 내부에서 관리
        video_task = db.query(VideoTask).filter(VideoTask.req_id == req_id).first()
        if video_task:
            video_task.status = config.TASK_STAT_CANCELED
            video_task.updated_at = datetime.now(timezone.utc)  # 업데이트 시간 갱신
            db.commit()  # 변경 사항 저장
            return {"message": f"Task req.id {req_id} has been {config.TASK_STAT_CANCELED}."}
        else:
            return {"message": f"Task req.id {req_id} not found."}


def is_task_cancelled(task_id: int):
    with SessionLocal() as db:
        video_task = db.query(VideoTask).filter(VideoTask.id == task_id).first()
        if video_task and video_task.status == config.TASK_STAT_CANCELED:
            return True
    return False

def create_video_task(username: str, req_id: str, video_url: str, process_options: list):
    """유저 정보를 바탕으로 VideoTask와 관련된 작업 타입을 생성하는 함수"""
    
    # 세션을 함수 내부에서 생성 및 관리
    with SessionLocal() as db:
        # 유저 정보 조회
        user = db.query(User).filter(User.username == username).first()
        if not user:
            return {"message": "User not found"}

        # VideoTask 생성
        video_task = VideoTask(
            user_id=user.id,
            req_id=req_id,
            video_url=video_url,
            status=config.TASK_STAT_CREATED,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        db.add(video_task)
        db.commit()  # task id 생성
        db.refresh(video_task)

        # 선택된 process_options에 따라 VideoTaskType 생성
        for option in process_options:
            task_type = VideoTaskType(
                task_id=video_task.id,
                result_type=option,
                created_at=datetime.now(timezone.utc)
            )
            db.add(task_type)
        db.commit()

        print(f"task is inserted, video task id: {video_task.id}")
        return video_task


def update_task_status_to_downloaded(task_id: int):
    with SessionLocal() as db:
        video_task = db.query(VideoTask).filter(VideoTask.id == task_id).first()
        if video_task:
            video_task.status = config.TASK_STAT_DOWNLOADED  
            video_task.updated_at = datetime.now(timezone.utc)  # 업데이트 시간 갱신
            db.commit()  # 변경 사항 저장
            return {"message": f"Task {task_id} updated to {config.TASK_STAT_DOWNLOADED}."}
        return {"message": f"Task {task_id} not found."}
    
    
def update_task_status_to_complete(task_id: int):
    with SessionLocal() as db:
        # video_task 인스턴스를 다시 세션에 바인딩하기 위해 merge 사용
        video_task = db.query(VideoTask).filter(VideoTask.id == task_id).first()
        if video_task:
            video_task = db.merge(video_task)  # 인스턴스를 세션에 다시 바인딩
            video_task.status = config.TASK_STAT_COMPLETED  # 상태를 'complete'로 업데이트
            video_task.updated_at = datetime.now(timezone.utc)  # 업데이트 시간 갱신
            db.commit()  # 변경 사항 저장
            return {"message": f"Task {task_id} updated to {config.TASK_STAT_COMPLETED}."}
        return {"message": f"Task {task_id} not found."}

def get_user_tasks(auth_state):
    username = auth_state.get("username")  # auth_state에서 username 가져오기
    
    if not username:
        return "로그인 정보가 없습니다."
    
    # 데이터베이스 세션을 열고 작업을 수행 후 자동으로 닫음
    with SessionLocal() as db:
        # User 테이블에서 username으로 사용자를 찾고 그 사용자의 ID를 조회
        user = db.query(User).filter(User.username == username).first()
        
        if not user:
            return "사용자를 찾을 수 없습니다."

        # 해당 사용자의 ID로 VideoTask 필터링
        tasks = db.query(VideoTask).options(joinedload(VideoTask.task_types)).filter(VideoTask.user_id == user.id).all()

    # 작업 목록을 리스트로 변환하여 반환
    task_list = []
    if tasks:
        for task in tasks:
            # 해당 작업에 연결된 VideoTaskType 가져오기
            task_types = ", ".join([task_type.result_type for task_type in task.task_types])
            task_list.append([
                task.id, task.status, task.video_url, 
                format_datetime_short(convert_to_kst(task.created_at)), 
                format_datetime_short(convert_to_kst(task.updated_at)), 
                task_types
            ])
    
    # DataFrame 생성 및 빈 경우의 처리
    df = pd.DataFrame(task_list, columns=["Task ID", "Status", "Video URL", "Created", "Updated", "Task Types"])
    
    if df.empty:
        # 데이터가 비어 있을 경우 빈 DataFrame 유지
        return df

    # created_at 기준으로 내림차순 정렬
    df = df.sort_values(by=["Created", "Task ID"], ascending=[False, True])
        
    return df


def get_task_by_req_id(req_id: str):
    with SessionLocal() as db:
        task = db.query(VideoTask).filter(VideoTask.req_id == req_id).first()
        return task


def create_video_result(task_id, file_path, result_type):
    with SessionLocal() as db:
        video_result = VideoResult(
            task_id=task_id,
            file_path=file_path,
            result_type=result_type
        )
        db.add(video_result)
        db.commit()

def get_video_results_by_task_id(task_id: int):
    with SessionLocal() as session:  # 올바른 세션 생성
        video_results = session.query(VideoResult).filter(VideoResult.task_id == task_id).all()
    
    return video_results

def get_video_task_by_req_id(req_id: str):
    with SessionLocal() as db:
        video_task = db.query(VideoTask).filter(VideoTask.req_id == req_id).first()
        
        if not video_task:
            raise HTTPException(status_code=404, detail="Task not found.")
        
        return video_task
    
def get_video_task_by_id(task_id: int):
    with SessionLocal() as db:
        video_task = db.query(VideoTask).filter(VideoTask.id == task_id).first()
        
        if not video_task:
            raise HTTPException(status_code=404, detail="Task not found.")
        
        return video_task


def get_video_result_by_task_id_and_type(task_id: int, video_type: str):
    with SessionLocal() as db:
        # 주어진 task_id와 video_type을 기준으로 VideoResult 조회
        result = db.query(VideoResult).filter(
            VideoResult.task_id == task_id,
            VideoResult.result_type == video_type
        ).first()
        
        if result:
            return result.file_path  # 결과 파일 경로 반환
        else:
            raise HTTPException(status_code=404, detail=f"{video_type} result not found for task_id {task_id}")


def update_task_status_to_error(task_id: int):
    with SessionLocal() as db:
        task = db.query(VideoTask).filter(VideoTask.id == task_id).first()
        
        if task:
            task.status = config.TASK_STAT_ERROR
            task.updated_at = datetime.now(timezone.utc)  # 업데이트 시간 갱신
            db.commit()  # 변경사항을 커밋하여 DB에 저장
        else:
            raise ValueError(f"Task with id {task_id} not found")
