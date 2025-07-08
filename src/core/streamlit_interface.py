import streamlit as st
import streamlit.components.v1 as components
import base64
import json
from datetime import datetime
from ui.components import create_video_player_html

def get_video_base64_from_path(video_path):
    """Convert local video file to base64 for embedding"""
    try:
        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()
            video_base64 = base64.b64encode(video_bytes).decode()
        return video_base64
    except Exception as e:
        st.error(f"Error reading video file: {e}")
        return None

def get_video_base64_from_upload(video_file):
    """Convert uploaded video to base64 for embedding"""
    video_file.seek(0)
    video_bytes = video_file.read()
    video_base64 = base64.b64encode(video_bytes).decode()
    return video_base64



def main():
    st.set_page_config(page_title="Real-Time Video Elicitation", layout="wide")
    
    # Initialize session state
    if 'current_time' not in st.session_state:
        st.session_state.current_time = 0
    if 'video_duration' not in st.session_state:
        st.session_state.video_duration = 0
    if 'elicitation_sessions' not in st.session_state:
        st.session_state.elicitation_sessions = []
    
    st.title("🎬 Real-Time Video Elicitation Interface")
    
    # Video upload
    uploaded_file = st.file_uploader(
        "Upload your video",
        type=['mp4', 'avi', 'mov', 'webm'],
        help="Upload a video file for elicitation"
    )
    
    if uploaded_file is not None:
        # Convert video to base64
        with st.spinner("Loading video..."):
            video_base64 = get_video_base64_from_path(uploaded_file)
            video_type = uploaded_file.type.split('/')[-1]
        
        st.success(f"✅ Video loaded: {{uploaded_file.name}}")
        
        # Create and display the video player
        player_html = create_video_player_html(video_base64, video_type)
        
        # Display the video player component
        video_state = components.html(
            player_html,
            height=600,
            scrolling=False
        )
        
        # Display current state information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Video State")
            st.write(f"**Current Time:** {{st.session_state.current_time:.1f}}s")
            st.write(f"**Duration:** {{st.session_state.video_duration:.1f}}s")
        
        with col2:
            st.subheader("📝 Elicitation Sessions")
            if st.session_state.elicitation_sessions:
                for session in st.session_state.elicitation_sessions:
                    with st.expander(f"Session {{session['id']}}"):
                        st.write(f"**Start:** {{session['startTime']:.1f}}s")
                        st.write(f"**End:** {{session['endTime']:.1f}}s")
                        st.write(f"**Duration:** {{session['duration']:.1f}}s")
            else:
                st.info("No elicitation sessions recorded yet")
    
    else:
        st.info("👆 Please upload a video file to begin")

if __name__ == "__main__":
    main()