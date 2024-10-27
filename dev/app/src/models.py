from pydantic import BaseModel

class VideoProcessInput(BaseModel):
    youtube_url: str
    start_time: str
    end_time: str