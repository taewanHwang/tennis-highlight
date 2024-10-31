from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import config

DATABASE_URL = f"mysql+pymysql://{config.MYSQL_DB_USER}:{config.MYSQL_DB_PWD}@{config.MYSQL_CONT_NAME}/{config.MYSQL_DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
