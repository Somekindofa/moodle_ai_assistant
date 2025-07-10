import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # Go up 2 levels
sys.path.insert(0, parent_dir)
import streamlit as st
import streamlit.components.v1 as components
import json
from datetime import datetime
import time
import threading
import http.server
import socketserver
import socket
from urllib.parse import quote
from src.ui.components import create_streaming_video_player_html

def find_free_port():
    """Find a free port for the HTTP server"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def start_video_server(video_path, port):
    """
    Start a simple HTTP server to serve video files with CORS support.
    
    This function creates and starts an HTTP server that serves video files from a specified
    directory.
    
    The server includes CORS headers to allow cross-origin access, making it
    suitable for web applications that need to access video content from different domains.
    
    Parameters
    ----------
    video_path : str
        The full path to the video file to be served. The server will serve files from
        the directory containing this video file.
    port : int
        The port number on which the HTTP server will listen for incoming requests.
        Must be a valid port number (typically 1024-65535 for non-privileged users).
    
    Returns
    -------
    None
        This function runs indefinitely until manually stopped or an error occurs.
    
    Raises
    ------
    Exception
        Catches and prints any server-related errors that occur during startup or
        operation, such as port already in use or permission denied.
    
    Notes
    -----
    - The server serves files from the directory containing the specified video file
    - CORS headers are automatically added to all responses
    - The server runs in a blocking manner until manually terminated
    - Uses Python's built-in http.server and socketserver modules
    Examples
    --------
    >>> start_video_server("/path/to/video.mp4", 8080)
    # Starts server on port 8080 serving files from /path/to/ directory
    """
    
    class VideoHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=os.path.dirname(video_path), **kwargs)
        
        def end_headers(self):
            # Add CORS headers for cross-origin access
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            super().end_headers()
    
    try:
        with socketserver.TCPServer(("", port), VideoHandler) as httpd:
            httpd.serve_forever()
    except Exception as e:
        print(f"Server error: {e}")


def main():
    st.set_page_config(page_title="Real-Time Video Elicitation", layout="wide")
    
    # Initialize session state
    if 'current_time' not in st.session_state:
        st.session_state.current_time = 0
    if 'video_duration' not in st.session_state:
        st.session_state.video_duration = 0
    if 'elicitation_sessions' not in st.session_state:
        st.session_state.elicitation_sessions = []
    if 'video_server_port' not in st.session_state:
        st.session_state.video_server_port = None
    if 'server_thread' not in st.session_state:
        st.session_state.server_thread = None
        st.title("🎬 Real-Time Video Elicitation Interface")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("📁 Video Selection")
        
        video_path = st.text_input(
            "Video File Path:",
            placeholder="C:/path/to/your/large_video.mp4",
            help="Enter the full path to your video file"
        )
        
        load_button = st.button(
            "Load Video", 
            type="primary", 
            use_container_width=True,
            icon="🎬"
        )
        
        # File browser helper
        with st.expander("💡 Example Paths"):
            st.code("C:\\Users\\YourName\\Videos\\video.mp4", language="text")
            st.code("/Users/YourName/Movies/video.mp4", language="text") 
            st.code("/home/username/videos/video.mp4", language="text")
        
        st.divider()
        
        # Video info section (will populate after loading)
        if 'current_video_info' in st.session_state:
            st.header("📊 Video Info")
            info = st.session_state.current_video_info
            st.write(f"**File:** {info['name']}")
            st.write(f"**Size:** {info['size']:.2f} GB")
            if st.session_state.video_duration > 0:
                st.write(f"**Duration:** {st.session_state.video_duration:.1f}s")
                st.write(f"**Current:** {st.session_state.current_time:.1f}s")
        
        st.divider()
        
        # Sessions summary in sidebar
        st.header("📝 Sessions")
        session_count = len(st.session_state.elicitation_sessions)
        st.metric("Total Sessions", session_count)
        
        if session_count > 0:
            total_duration = sum(s['duration'] for s in st.session_state.elicitation_sessions)
            st.metric("Total Duration", f"{total_duration:.1f}s")
    
    if video_path and load_button:
        if os.path.exists(video_path):
            try:                # Get file info
                file_size = os.path.getsize(video_path) / (1024 * 1024 * 1024)  # GB
                file_name = os.path.basename(video_path)
                
                # Store in session state for sidebar
                st.session_state.current_video_info = {
                    'name': file_name,
                    'size': file_size,
                    'path': video_path
                }
                
                st.success(f"✅ **{file_name}** ({file_size:.2f} GB)")
                
                # Stop existing server if running
                if st.session_state.server_thread and st.session_state.server_thread.is_alive():
                    # Note: In production, you'd want proper server shutdown
                    pass
                
                # Start new HTTP server
                port = find_free_port()
                st.session_state.video_server_port = port
                
                # Start server in background thread
                server_thread = threading.Thread(
                    target=start_video_server, 
                    args=(video_path, port),
                    daemon=True
                )
                server_thread.start()
                st.session_state.server_thread = server_thread
                
                # Give server time to start
                time.sleep(0.5)
                
                # Create video URL
                video_filename = quote(os.path.basename(video_path))
                video_url = f"http://localhost:{port}/{video_filename}"
                
                st.info(f"🌐 **Streaming from:** `localhost:{port}`")
                
                # Create and display the video player
                player_html = create_streaming_video_player_html(video_url)
                  # Display the video player component
                components.html(
                    player_html,
                    height=700,
                    scrolling=False
                )
                
                # Sessions detail below video (optional)
                if st.session_state.elicitation_sessions:
                    with st.expander("📝 Recent Session Details"):
                        for session in st.session_state.elicitation_sessions[-5:]:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**Session {session['id']}**")
                            with col2:
                                st.write(f"{session['startTime']:.1f}s → {session['endTime']:.1f}s")
                            with col3:
                                st.write(f"Duration: {session['duration']:.1f}s")
                  # Instructions
                with st.expander("📖 How to Use"):
                    st.markdown("""
                    **Recording Elicitations:**
                    1. 🔴 Click "Record" to start capturing an elicitation
                    2. The video will automatically play during recording
                    3. ⏹️ Click "Stop Recording" to finish and save the session
                    
                    **Navigation:**
                    - Use the timeline slider to scrub through the video
                    - Click on session markers (green bars) to jump to recorded segments
                    - Sessions are displayed below with clickable timestamps
                    
                    **Performance:**
                    - Videos stream directly from your disk (no memory limits)
                    - Smooth playback even for multi-GB files
                    - Sessions capture precise temporal windows for Whisper processing
                    """)
                
            except Exception as e:
                st.error(f"❌ Error setting up video: {e}")
        else:
            st.error("❌ File path does not exist. Please check the path and try again.")
    
    elif video_path and not load_button:
        st.info("👆 Click 'Load Video' to start streaming")
    else:
        st.info("👆 Enter the path to your video file")
        
        # Example paths for different operating systems
        with st.expander("💡 Example Paths"):
            st.code("Windows: C:\\Users\\YourName\\Videos\\video.mp4")
            st.code("macOS: /Users/YourName/Movies/video.mp4") 
            st.code("Linux: /home/username/videos/video.mp4")

if __name__ == "__main__":
    main()
