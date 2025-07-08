def create_video_player_html(video_base64, video_type="mp4"):
    """Create HTML video player with JavaScript controls"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .video-container {{
                width: 100%;
                max-width: 800px;
                margin: 0 auto;
            }}
            
            .video-player {{
                width: 100%;
                background: #000;
                border-radius: 8px;
            }}
            
            .controls {{
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 10px;
                background: #f0f0f0;
                border-radius: 0 0 8px 8px;
            }}
            
            .timeline {{
                flex: 1;
                height: 6px;
                background: #ddd;
                border-radius: 3px;
                position: relative;
                cursor: pointer;
            }}
            
            .timeline-progress {{
                height: 100%;
                background: #007bff;
                border-radius: 3px;
                width: 0%;
            }}
            
            .timeline-handle {{
                position: absolute;
                top: -5px;
                width: 16px;
                height: 16px;
                background: #007bff;
                border-radius: 50%;
                cursor: pointer;
                left: 0%;
                transform: translateX(-50%);
            }}
            
            .time-display {{
                font-family: monospace;
                font-size: 14px;
                min-width: 120px;
            }}
            
            .btn {{
                padding: 8px 12px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
            }}
            
            .btn-primary {{
                background: #007bff;
                color: white;
            }}
            
            .btn-secondary {{
                background: #6c757d;
                color: white;
            }}
            
            .btn-danger {{
                background: #dc3545;
                color: white;
            }}
            
            .recording-indicator {{
                padding: 10px;
                background: #dc3545;
                color: white;
                border-radius: 4px;
                text-align: center;
                margin: 10px 0;
                display: none;
            }}
            
            .recording-indicator.active {{
                display: block;
            }}
            
            .elicitation-sessions {{
                margin-top: 20px;
                max-height: 300px;
                overflow-y: auto;
            }}
            
            .session {{
                background: #f8f9fa;
                padding: 10px;
                margin: 5px 0;
                border-radius: 4px;
                border-left: 4px solid #28a745;
            }}
        </style>
    </head>
    <body>
        <div class="video-container">
            <video id="videoPlayer" class="video-player" controls>
                <source src="data:video/{video_type};base64,{video_base64}" type="video/{video_type}">
                Your browser does not support the video tag.
            </video>
            
            <div class="controls">
                <button id="playBtn" class="btn btn-primary">▶️ Play</button>
                <button id="pauseBtn" class="btn btn-secondary">⏸️ Pause</button>
                
                <div class="timeline" id="timeline">
                    <div class="timeline-progress" id="progress"></div>
                    <div class="timeline-handle" id="handle"></div>
                </div>
                
                <div class="time-display" id="timeDisplay">00:00 / 00:00</div>
                
                <button id="recordBtn" class="btn btn-danger">🔴 Record</button>
            </div>
            
            <div id="recordingIndicator" class="recording-indicator">
                <strong>🔴 RECORDING IN PROGRESS</strong><br>
                Duration: <span id="recordingDuration">00:00</span>
            </div>
            
            <div class="elicitation-sessions">
                <h3>Elicitation Sessions</h3>
                <div id="sessionsList"></div>
            </div>
        </div>

        <script>
            const video = document.getElementById('videoPlayer');
            const playBtn = document.getElementById('playBtn');
            const pauseBtn = document.getElementById('pauseBtn');
            const timeline = document.getElementById('timeline');
            const progress = document.getElementById('progress');
            const handle = document.getElementById('handle');
            const timeDisplay = document.getElementById('timeDisplay');
            const recordBtn = document.getElementById('recordBtn');
            const recordingIndicator = document.getElementById('recordingIndicator');
            const recordingDuration = document.getElementById('recordingDuration');
            const sessionsList = document.getElementById('sessionsList');
            
            let isRecording = false;
            let recordingStartTime = null;
            let elicitationSessions = [];
            let isDragging = false;
            
            // Format time helper
            function formatTime(seconds) {{
                const mins = Math.floor(seconds / 60);
                const secs = Math.floor(seconds % 60);
                return `${{mins.toString().padStart(2, '0')}}:${{secs.toString().padStart(2, '0')}}`;
            }}
            
            // Update timeline and time display
            function updateTimeline() {{
                if (video.duration && !isDragging) {{
                    const percent = (video.currentTime / video.duration) * 100;
                    progress.style.width = percent + '%';
                    handle.style.left = percent + '%';
                    
                    timeDisplay.textContent = 
                        formatTime(video.currentTime) + ' / ' + formatTime(video.duration);
                    
                    // Send current time to Streamlit
                    window.parent.postMessage({{
                        type: 'video_time_update',
                        currentTime: video.currentTime,
                        duration: video.duration,
                        isPlaying: !video.paused,
                        isRecording: isRecording
                    }}, '*');
                    
                    // Update recording duration if recording
                    if (isRecording && recordingStartTime !== null) {{
                        const recordDuration = video.currentTime - recordingStartTime;
                        recordingDuration.textContent = formatTime(recordDuration);
                    }}
                }}
            }}
            
            // Event listeners
            playBtn.addEventListener('click', () => {{
                video.play();
            }});
            
            pauseBtn.addEventListener('click', () => {{
                video.pause();
            }});
            
            // Timeline scrubbing
            timeline.addEventListener('mousedown', (e) => {{
                isDragging = true;
                updateVideoTime(e);
            }});
            
            document.addEventListener('mousemove', (e) => {{
                if (isDragging) {{
                    updateVideoTime(e);
                }}
            }});
            
            document.addEventListener('mouseup', () => {{
                isDragging = false;
            }});
            
            function updateVideoTime(e) {{
                const rect = timeline.getBoundingClientRect();
                const percent = (e.clientX - rect.left) / rect.width;
                const newTime = percent * video.duration;
                video.currentTime = Math.max(0, Math.min(newTime, video.duration));
                updateTimeline();
            }}
            
            // Recording functionality
            recordBtn.addEventListener('click', () => {{
                if (!isRecording) {{
                    // Start recording
                    isRecording = true;
                    recordingStartTime = video.currentTime;
                    recordBtn.textContent = '⏹️ Stop';
                    recordBtn.className = 'btn btn-secondary';
                    recordingIndicator.classList.add('active');
                    
                    // Auto-play when recording starts
                    video.play();
                }} else {{
                    // Stop recording
                    const endTime = video.currentTime;
                    const session = {{
                        id: elicitationSessions.length + 1,
                        startTime: recordingStartTime,
                        endTime: endTime,
                        duration: endTime - recordingStartTime,
                        timestamp: new Date().toLocaleString()
                    }};
                    
                    elicitationSessions.push(session);
                    displaySession(session);
                    
                    // Send session to Streamlit
                    window.parent.postMessage({{
                        type: 'elicitation_session',
                        session: session
                    }}, '*');
                    
                    // Reset recording state
                    isRecording = false;
                    recordingStartTime = null;
                    recordBtn.textContent = '🔴 Record';
                    recordBtn.className = 'btn btn-danger';
                    recordingIndicator.classList.remove('active');
                    
                    video.pause();
                }}
            }});
            
            // Display elicitation session
            function displaySession(session) {{
                const sessionDiv = document.createElement('div');
                sessionDiv.className = 'session';
                sessionDiv.innerHTML = `
                    <h4>Session ${{session.id}}</h4>
                    <p><strong>Time:</strong> ${{formatTime(session.startTime)}} - ${{formatTime(session.endTime)}}</p>
                    <p><strong>Duration:</strong> ${{formatTime(session.duration)}}</p>
                    <p><strong>Timestamp:</strong> ${{session.timestamp}}</p>
                `;
                sessionsList.appendChild(sessionDiv);
            }}
            
            // Listen for messages from Streamlit
            window.addEventListener('message', (event) => {{
                if (event.data.type === 'seek_video') {{
                    video.currentTime = event.data.time;
                    updateTimeline();
                }}
            }});
            
            // Update timeline continuously
            video.addEventListener('timeupdate', updateTimeline);
            video.addEventListener('loadedmetadata', updateTimeline);
            
            // Initialize
            video.addEventListener('loadedmetadata', () => {{
                updateTimeline();
                // Send initial state to Streamlit
                window.parent.postMessage({{
                    type: 'video_loaded',
                    duration: video.duration
                }}, '*');
            }});
        </script>
    </body>
    </html>
    """