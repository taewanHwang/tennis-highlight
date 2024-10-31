import gradio as gr
from gradio_utils import auth, show_task_list, init_tasks_func, init_tasks_func, update_tasks_func, show_new_task, update_tasks_func, on_row_select, start_download_full, start_download_playing, start_download_highlights, start_download_segments, update_ui, start_processing, stop_processing, refresh_status
import config

##################################### Main #####################################

# Main app view to toggle between task list and task detail
with gr.Blocks(title="Tennis highlight editor") as main_view:
    task_list_container = gr.Group(visible=True)
    task_detail_container = gr.Group(visible=False) 
    new_task_container = gr.Group(visible=False) 
    
    auth_state = gr.State({"username": None})
    task_state = gr.State({"created_task_id": None, "detail_task_id":None})    


    with task_detail_container:
        back_button = gr.Button("◀ Back")
        task_id_text = gr.Markdown()  # Placeholder for Task ID   
        
        download_full_button = gr.Button("⭐ Download Full Video ⭐", interactive=False)
        download_playing_button = gr.Button("⭐ Download Playing Video ⭐", interactive=False)
        download_highlights_button = gr.Button(f"⭐ Download {config.HIGHTLIGHTS_NUM} Highlights Video ⭐", interactive=False)
        download_segments_button = gr.Button("⭐ Download Segments Video ⭐", interactive=False)
        download_output = gr.File(label="⭐ Download Result Video ⭐", interactive=False)

        back_button.click(show_task_list, inputs=[], outputs=[task_list_container, task_detail_container, new_task_container])
        download_full_button.click(start_download_full, inputs=task_state, outputs=download_output)
        download_playing_button.click(start_download_playing, inputs=task_state, outputs=download_output)
        download_highlights_button.click(start_download_highlights, inputs=task_state, outputs=download_output)
        download_segments_button.click(start_download_segments, inputs=task_state, outputs=download_output)

    with task_list_container:
        welcome_markdown = gr.Markdown()  # 사용자 이름을 표시할 마크다운 컴포넌트

        new_tasks_button = gr.Button("✅ New Task")
        update_tasks_button = gr.Button("🔄 Update Tasks")
        task_table = gr.DataFrame(headers=["Task ID", "Status", "Video URL", "Created", "Updated","Task Types"])  # 테이블 컴포넌트


        main_view.load(init_tasks_func, inputs=None, outputs=[auth_state, task_table, welcome_markdown])
        new_tasks_button.click(show_new_task, inputs=[], outputs=[task_list_container, task_detail_container, new_task_container])
        update_tasks_button.click(update_tasks_func, inputs=[], outputs=[auth_state, task_table])

        task_table.select(on_row_select, inputs=[task_state], outputs=[task_list_container, task_detail_container, new_task_container, download_full_button, download_playing_button, download_highlights_button, download_segments_button, task_state])  


    with new_task_container:
        back_button = gr.Button("◀ Back")
        back_button.click(show_task_list, inputs=[], outputs=[task_list_container, task_detail_container, new_task_container])

        # 입력 방식 선택 (파일 업로드 또는 유튜브 링크 입력)
        input_method = gr.Dropdown(["YouTube Link", "File Upload"], label="Choose input method", value="YouTube Link")
        
        # 파일 업로드 관련 필드
        upload_file = gr.File(label="Upload Video File", visible=False)
        
        # 유튜브 링크 및 시간 입력 관련 필드 (처음에는 보이지 않음)
        youtube_link = gr.Textbox(label="YouTube Video Link", visible=True)
        start_time = gr.Textbox(label="Start Time (HH:MM:SS)", value="00:00:00", visible=True)
        end_time = gr.Textbox(label="End Time (HH:MM:SS)", value="00:01:00", visible=True)
        
        # 프로세스 옵션 선택
        process_options = gr.CheckboxGroup(
            choices=["Full", "Playing", "Highlight"],
            label="Select Processing Options",
            value=["Playing", "Highlight"],
            type="value",
            show_label=True,
            elem_id="checkbox-row"
        )


        process_button = gr.Button("🚀 Start Processing")
        stop_button = gr.Button("🚫 Stop Processing", interactive=True)  # 작업 중지 버튼 추가

        # 상태 확인 및 다운로드 버튼
        status_output = gr.Textbox(label="Status", lines=2)
        refresh_button = gr.Button("🔄 Update Status")
                
        # 입력 방식 변경에 따른 UI 업데이트
        input_method.change(update_ui, inputs=input_method, outputs=[upload_file, youtube_link, start_time, end_time])

        # 이벤트 핸들러 설정
        process_button.click(start_processing, 
                            inputs=[upload_file, youtube_link, start_time, end_time, input_method, process_options, task_state, auth_state], 
                            outputs=[status_output, task_state])

        stop_button.click(stop_processing, inputs=task_state, outputs=[status_output]) 
        refresh_button.click(refresh_status, inputs=task_state, outputs=[status_output])

if __name__ == "__main__":
    main_view.launch(auth=auth, share=False, server_port=9000, server_name="0.0.0.0")
   
    print("gradio app start", flush=True)
