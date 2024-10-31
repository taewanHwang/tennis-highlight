from db import Base
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)  # 비밀번호 해시 저장
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

    tasks = relationship("VideoTask", back_populates="user")  # 사용자와 작업의 관계


class VideoTask(Base):
    __tablename__ = 'video_tasks'

    id = Column(Integer, primary_key=True, index=True)
    req_id = Column(String(255), nullable=False) # 
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    video_url = Column(String(255), nullable=False)  # 비디오 파일 경로나 유튜브 URL
    status = Column(String(50), default='pending')  # 작업 상태 (pending, processing, completed, failed)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))

    user = relationship("User", back_populates="tasks")
    task_types = relationship("VideoTaskType", back_populates="task")  # 연관 테이블을 통한 결과물 관리
    results = relationship("VideoResult", back_populates="task")  # 처리된 결과물과의 관계


class VideoTaskType(Base):
    __tablename__ = 'video_task_types'

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey('video_tasks.id'), nullable=False)
    result_type = Column(String(50), nullable=False)  # full, playing, highlight, highlights_zip, segments_zip 등
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

    task = relationship("VideoTask", back_populates="task_types")


class VideoResult(Base):
    __tablename__ = 'video_results'

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey('video_tasks.id'), nullable=False)  # 해당 작업과 연관
    file_path = Column(String(255), nullable=False)  # 결과 파일 경로
    result_type = Column(String(50), nullable=False)  # full, playing, highlight, highlights_zip, segments_zip 등
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

    task = relationship("VideoTask", back_populates="results")
