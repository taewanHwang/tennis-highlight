from pydantic import BaseModel

class YouTubeVideoInput(BaseModel):
    youtube_url: str
    start_time: str
    end_time: str
    process_options: list[str]
    username: str
