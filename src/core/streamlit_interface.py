import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Video Elicitation Interface",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    """Load custom CSS from st_MD.css file"""
    css_path = "../../css/st_MD.css"
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"""
        <style>
        {css_content}
        </style>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found at {css_path}")
    except Exception as e:
        st.error(f"Error loading CSS: {e}")

load_css()

def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'current_time': 0,
        'video_duration': 300,  # 5 minutes
        'is_playing': False,
        'is_recording': False,
        'user_mode': 'expert',
        'elicitation_start_time': None,
        'elicitation_sessions': [],
        'auto_play_timer': None,
        'last_update_time': None
    }
    
    for key, value in defaults.items():
        # pushing the initial state
        if key not in st.session_state:
            st.session_state[key] = value

def format_time(seconds):
    """Convert seconds to MM:SS format"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"

def create_temporal_window(start_time, end_time, buffer_seconds=5):
    """Create temporal window with buffer for video segment extraction"""
    window_start = max(0, start_time - buffer_seconds)
    window_end = min(st.session_state.video_duration, end_time + buffer_seconds)
    return {
        'original_start': start_time,
        'original_end': end_time,
        'window_start': window_start,
        'window_end': window_end,
        'duration': window_end - window_start
    }

def simulate_whisper_transcription(duration):
    """Simulate Whisper transcription (replace with real API call)"""
    sample_phrases = [
        "The hand position is crucial for safety and precision.",
        "Notice how the cutting angle affects the wood grain.",
        "This technique requires steady pressure and control.",
        "The tool alignment is essential for clean cuts.",
        "Pay attention to the wood fiber direction."
    ]
    return np.random.choice(sample_phrases)

def handle_elicitation_start():
    """Handle the start of elicitation recording"""
    st.session_state.is_recording = True
    st.session_state.elicitation_start_time = st.session_state.current_time
    st.session_state.is_playing = True
    st.session_state.last_update_time = time.time()

def handle_elicitation_stop():
    """Handle the stop of elicitation recording"""
    if st.session_state.elicitation_start_time is not None:
        end_time = st.session_state.current_time
        start_time = st.session_state.elicitation_start_time
        
        # Create temporal window
        temporal_window = create_temporal_window(start_time, end_time)
        
        # Simulate transcription
        transcription = simulate_whisper_transcription(temporal_window['duration'])
        
        # Create elicitation session record
        session = {
            'id': len(st.session_state.elicitation_sessions) + 1,
            'timestamp': datetime.now(),
            'start_time': start_time,
            'end_time': end_time,
            'temporal_window': temporal_window,
            'transcription': transcription,
            'status': 'completed'
        }
        
        st.session_state.elicitation_sessions.append(session)
    
    st.session_state.is_recording = False
    st.session_state.elicitation_start_time = None
    st.session_state.is_playing = False

def update_video_time():
    """Update video time when playing"""
    if st.session_state.is_playing and st.session_state.last_update_time:
        current_real_time = time.time()
        elapsed = current_real_time - st.session_state.last_update_time
        new_time = min(st.session_state.current_time + elapsed, st.session_state.video_duration)
        
        if new_time >= st.session_state.video_duration:
            st.session_state.is_playing = False
        
        st.session_state.current_time = new_time
        st.session_state.last_update_time = current_real_time

def main():
    init_session_state()
    
    # Update video time if playing
    if st.session_state.is_playing:
        update_video_time()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>Video Elicitation Interface</h1>
        <p>Expert Knowledge Capture System for Video Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        
        # User mode selection
        user_mode = st.radio(
            "User Mode",
            ["Expert", "Student"],
            index=0 if st.session_state.user_mode == 'expert' else 1
        )
        st.session_state.user_mode = user_mode.lower()
        
        st.divider()
        
        # Video controls
        st.subheader("Video Player")
        
        # Play/Pause buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Play", disabled=st.session_state.is_recording):
                st.session_state.is_playing = True
                st.session_state.last_update_time = time.time()
        
        with col2:
            if st.button("Pause"):
                st.session_state.is_playing = False
        
        # Time display
        st.write(f"Time: {format_time(st.session_state.current_time)} / {format_time(st.session_state.video_duration)}")
        
        st.divider()
        
        # Elicitation sessions list
        if st.session_state.elicitation_sessions:
            st.subheader("Elicitation Sessions")
            for session in st.session_state.elicitation_sessions:
                with st.expander(f"Session {session['id']} - {format_time(session['start_time'])}"):
                    st.write(f"**Duration:** {format_time(session['end_time'] - session['start_time'])}")
                    st.write(f"**Window:** {format_time(session['temporal_window']['window_start'])} - {format_time(session['temporal_window']['window_end'])}")
                    st.write(f"**Transcription:** {session['transcription']}")

    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Video player simulation
        st.subheader("Video Player - Carpentry Learning")
        
        # Video container
        video_status = "PLAYING" if st.session_state.is_playing else "PAUSED"
        recording_status = " | RECORDING" if st.session_state.is_recording else ""
        
        st.markdown(f"""
        <div class="video-container">
            <h2>Video Player</h2>
            <p>Ego-centric carpentry learning video</p>
            <p><strong>Current Time:</strong> {format_time(st.session_state.current_time)}</p>
            <p><strong>Status:</strong> {video_status}{recording_status}</p>
            <div style="background: #374151; padding: 1rem; margin: 1rem 0; border-radius: 5px;">
                <p>Simulation of carpentry video content</p>
                <p>First-person perspective of craftsman</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Video timeline (right under the video)
        st.subheader("Video Timeline")
        
        # Timeline slider
        new_time = st.slider(
            "Video Position",
            min_value=0,
            max_value=st.session_state.video_duration,
            value=int(st.session_state.current_time),
            step=1,
            disabled=st.session_state.is_recording,
            key="video_timeline"
        )
        
        # Update time if slider moved and not recording
        if not st.session_state.is_recording and new_time != int(st.session_state.current_time):
            st.session_state.current_time = float(new_time)
            st.session_state.is_playing = False
        
        # Timeline with elicitation markers
        timeline_html = f"""
        <div class="timeline-container">
            <div style="background: #e5e7eb; height: 20px; border-radius: 10px; position: relative;">
                <div style="background: #3b82f6; height: 20px; border-radius: 10px; width: {(st.session_state.current_time/st.session_state.video_duration)*100}%;"></div>
        """
        
        # Add elicitation session markers
        for session in st.session_state.elicitation_sessions:
            start_pos = (session['start_time'] / st.session_state.video_duration) * 100
            end_pos = (session['end_time'] / st.session_state.video_duration) * 100
            timeline_html += f"""
            <div style="position: absolute; left: {start_pos}%; top: 2px; width: {max(2, end_pos - start_pos)}%; height: 16px; 
                        background: #10b981; border: 1px solid white; border-radius: 3px; opacity: 0.8;" 
                 title="Elicitation {session['id']}: {format_time(session['start_time'])} - {format_time(session['end_time'])}"></div>
            """
        
        timeline_html += "</div></div>"
        st.markdown(timeline_html, unsafe_allow_html=True)

    with col2:
        # Elicitation controls (for experts only)
        if st.session_state.user_mode == 'expert':
            st.subheader("Elicitation Control")
            
            # Main elicitation button
            button_class = "elicitation-button recording-active" if st.session_state.is_recording else "elicitation-button"
            button_text = "STOP" if st.session_state.is_recording else "REC"
            
            st.markdown(f"""
            <div style="text-align: center;">
                <div class="{button_class}">
                    {button_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Button functionality
            col_start, col_stop = st.columns(2)
            with col_start:
                if st.button("Start Recording", disabled=st.session_state.is_recording, type="primary"):
                    handle_elicitation_start()
                    st.rerun()
            
            with col_stop:
                if st.button("Stop Recording", disabled=not st.session_state.is_recording, type="secondary"):
                    handle_elicitation_stop()
                    st.rerun()
            
            # Recording status
            if st.session_state.is_recording:
                st.markdown("""
                <div class="recording-indicator">
                    <strong>RECORDING IN PROGRESS</strong><br>
                    Video is playing automatically<br>
                    Speak now to elicit the video content
                </div>
                """, unsafe_allow_html=True)
                
                if st.session_state.elicitation_start_time is not None:
                    duration = st.session_state.current_time - st.session_state.elicitation_start_time
                    st.write(f"Recording duration: {format_time(duration)}")
        
        # Elicitation sessions display
        if st.session_state.elicitation_sessions:
            st.subheader("Captured Sessions")
            
            for session in reversed(st.session_state.elicitation_sessions[-3:]):  # Show last 3
                st.markdown(f"""
                <div class="elicitation-session">
                    <h5>Session {session['id']}</h5>
                    <p><strong>Time:</strong> {format_time(session['start_time'])} - {format_time(session['end_time'])}</p>
                    <p><strong>Transcription:</strong> {session['transcription']}</p>
                    <small>Window: {format_time(session['temporal_window']['window_start'])} - {format_time(session['temporal_window']['window_end'])}</small>
                </div>
                """, unsafe_allow_html=True)

    # Auto-refresh for video playing
    if st.session_state.is_playing or st.session_state.is_recording:
        time.sleep(0.1)
        st.rerun()

if __name__ == "__main__":
    main()