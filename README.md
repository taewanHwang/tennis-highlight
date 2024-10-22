# Tennis Video Processing Service

This project provides a web interface for processing tennis videos, either by uploading a file or by providing a YouTube link. The interface allows users to check the status of the processing tasks, download various versions of the processed video (full, playing, highlights, segments), and stop any ongoing tasks.

The service is built using Gradio for the UI, and FastAPI for handling video processing on the backend.

## Features

- **Video File Upload**: Upload your local video files for processing.
- **YouTube Link Input**: Provide a YouTube video link and specify the time range for processing.
- **Task Status Monitoring**: Check the status of video processing (e.g., Pending, Started, or Completed).
- **Task Management**: Option to stop ongoing tasks if needed.
- **Video Download**: Download the processed video in different formats: full, playing, highlights, segments.

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your_username/tennis-video-processing.git
cd tennis-video-processing
```

###  2. Create and activate a virtual environment
Create a Python virtual environment to isolate dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
Install the required Python packages from requirements.txt:
```bash
pip install -r requirements.txt
```

### 4. Start all
To restart the service (FastAPI and Gradio), use the restart.sh script:
```bash
./restart.sh
```


### 5. Etc
User Authentication
- User authentication is handled through a user list stored in a local file. You can manage users in the auth_users file. Only authenticated users can access the Gradio UI.
Scripts
- restart.sh: Restarts the backend (FastAPI) and Gradio services.
- run.sh: Displays logs for monitoring the backend and Gradio services.
Endpoints
- /process_video/: Upload and process a video file.
- /process_youtube_video/: Process a YouTube video with a specified start and end time.
- /check_status/{task_id}: Check the processing status of a task.
- /download_video/{task_id}: Download the processed video.
- /stop_task/{task_id}: Stop an ongoing video processing task.
Task Flow
- Select Input Method: Choose between File Upload or YouTube Link.
- Start Processing: Either upload a video file or input a YouTube video link and specify the start and end times. Click "Start Processing" to begin.
- Monitor Task Status: Use the "Refresh Status" button to check the current status of the processing task.
- Download Processed Video: Once processing is complete, download the processed video in different formats using the provided buttons.






