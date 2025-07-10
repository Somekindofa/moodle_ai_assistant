def create_streaming_video_player_html(video_url):
    """Create HTML video player that streams from local server"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .video-container {{
                width: 100%;
                max-width: 800px;
                margin: 0 auto;
                font-family: Arial, sans-serif;
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
                padding: 15px;
                background: #f8f9fa;
                border-radius: 0 0 8px 8px;
                border: 1px solid #dee2e6;
            }}
              .timeline {{
                flex: 1;
                height: 8px;
                background: #e9ecef;
                border-radius: 4px;
                position: relative;
                cursor: pointer;
                margin: 0 10px;
                transition: height 0.2s ease;
            }}
            
            .timeline:hover {{
                height: 12px;
            }}
            
            .timeline-progress {{
                height: 100%;
                background: linear-gradient(90deg, #007bff, #0056b3);
                border-radius: 4px;
                width: 0%;
                transition: width 0.1s ease;
            }}
            
            .timeline-handle {{
                position: absolute;
                top: -6px;
                width: 20px;
                height: 20px;
                background: #007bff;
                border: 3px solid white;
                border-radius: 50%;
                cursor: pointer;
                left: 0%;
                transform: translateX(-50%);
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }}
            
            .timeline-markers {{
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 100%;
                pointer-events: none;
            }}
            
            .session-marker {{
                position: absolute;
                top: -2px;
                height: 12px;
                background: #28a745;
                border-radius: 2px;
                opacity: 0.8;
                border: 1px solid white;
            }}
            
            .time-display {{
                font-family: 'Courier New', monospace;
                font-size: 14px;
                font-weight: bold;
                min-width: 140px;
                color: #495057;
                background: #e9ecef;
                padding: 5px 10px;
                border-radius: 4px;
            }}
            
            .btn {{
                padding: 10px 16px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
                gap: 5px;
            }}
            
            .btn:hover {{
                transform: translateY(-1px);
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            
            .btn-primary {{
                background: #007bff;
                color: white;
            }}
            
            .btn-primary:hover {{
                background: #0056b3;
            }}
            
            .btn-secondary {{
                background: #6c757d;
                color: white;
            }}
            
            .btn-secondary:hover {{
                background: #545b62;
            }}
            
            .btn-danger {{
                background: #dc3545;
                color: white;
            }}
            
            .btn-danger:hover {{
                background: #c82333;
            }}
            
            .recording-indicator {{
                padding: 15px;
                background: linear-gradient(45deg, #dc3545, #c82333);
                color: white;
                border-radius: 8px;
                text-align: center;
                margin: 15px 0;
                display: none;
                box-shadow: 0 4px 8px rgba(220, 53, 69, 0.3);
                animation: pulse 2s infinite;
            }}
            
            @keyframes pulse {{
                0% {{ box-shadow: 0 4px 8px rgba(220, 53, 69, 0.3); }}
                50% {{ box-shadow: 0 4px 20px rgba(220, 53, 69, 0.6); }}
                100% {{ box-shadow: 0 4px 8px rgba(220, 53, 69, 0.3); }}
            }}
            
            .recording-indicator.active {{
                display: block;
            }}
            
            .elicitation-sessions {{
                margin-top: 20px;
                max-height: 400px;
                overflow-y: auto;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                background: white;
            }}
            
            .sessions-header {{
                background: #f8f9fa;
                padding: 15px;
                border-bottom: 1px solid #dee2e6;
                font-weight: bold;
                color: #495057;
            }}
            
            .session {{
                background: white;
                padding: 15px;
                margin: 0;
                border-bottom: 1px solid #f1f3f4;
                border-left: 4px solid #28a745;
                transition: background 0.2s ease;
            }}
            
            .session:hover {{
                background: #f8f9fa;
            }}
            
            .session:last-child {{
                border-bottom: none;
                border-radius: 0 0 8px 8px;
            }}
            
            .session-title {{
                font-weight: bold;
                color: #495057;
                margin-bottom: 8px;
            }}
            
            .session-details {{
                color: #6c757d;
                font-size: 13px;
                line-height: 1.4;
            }}
            
            .video-info {{
                background: #e7f3ff;
                padding: 10px 15px;
                border-radius: 6px;
                margin-bottom: 15px;
                border: 1px solid #bee5eb;
                font-size: 13px;
                color: #0c5460;
            }}
            
            .loading-spinner {{
                display: inline-block;
                width: 16px;
                height: 16px;
                border: 2px solid #f3f3f3;
                border-top: 2px solid #007bff;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-right: 8px;
            }}
            
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
        </style>
    </head>
    <body>
        <div class="video-container">
            <div class="video-info">
                <strong>🎬 Streaming Video Player</strong> - Large files supported via HTTP streaming
            </div>
              <video id="videoPlayer" class="video-player" preload="metadata">
                <source src="{video_url}" type="video/mp4">
                Your browser does not support the video tag.
            </video><div class="controls">
                <button id="playBtn" class="btn btn-primary">
                    <span id="playIcon">▶️</span> <span id="playText">Play</span>
                </button>
                
                <div class="timeline" id="timeline">
                    <div class="timeline-progress" id="progress"></div>
                    <div class="timeline-markers" id="markers"></div>
                    <div class="timeline-handle" id="handle"></div>
                </div>
                
                <div class="time-display" id="timeDisplay">
                    <span class="loading-spinner"></span>Loading...
                </div>
                
                <button id="recordBtn" class="btn btn-danger">🔴 Record</button>
            </div>
            
            <div id="recordingIndicator" class="recording-indicator">
                <strong>🔴 RECORDING IN PROGRESS</strong><br>
                Duration: <span id="recordingDuration">00:00</span><br>
                <small>Video will auto-play during recording</small>
            </div>
            
            <div class="elicitation-sessions">
                <div class="sessions-header">📝 Elicitation Sessions</div>
                <div id="sessionsList">
                    <div style="padding: 20px; text-align: center; color: #6c757d; font-style: italic;">
                        No sessions recorded yet. Click "Record" to start capturing elicitations.
                    </div>
                </div>
            </div>
        </div>

        <script>            const video = document.getElementById('videoPlayer');
            const playBtn = document.getElementById('playBtn');
            const playIcon = document.getElementById('playIcon');
            const playText = document.getElementById('playText');
            const timeline = document.getElementById('timeline');
            const progress = document.getElementById('progress');
            const handle = document.getElementById('handle');
            const markers = document.getElementById('markers');
            const timeDisplay = document.getElementById('timeDisplay');
            const recordBtn = document.getElementById('recordBtn');
            const recordingIndicator = document.getElementById('recordingIndicator');
            const recordingDuration = document.getElementById('recordingDuration');
            const sessionsList = document.getElementById('sessionsList');
            
            let isRecording = false;
            let recordingStartTime = null;
            let elicitationSessions = [];
            let isDragging = false;
            let isLoaded = false;
            
            // Format time helper
            function formatTime(seconds) {{
                if (isNaN(seconds) || !isFinite(seconds)) return "00:00";
                const mins = Math.floor(seconds / 60);
                const secs = Math.floor(seconds % 60);
                return `${{mins.toString().padStart(2, '0')}}:${{secs.toString().padStart(2, '0')}}`;
            }}
              // Update timeline and time display
            function updateTimeline() {{
                if (video.duration && isLoaded) {{
                    const percent = (video.currentTime / video.duration) * 100;
                    
                    // Only update visuals if not dragging to prevent conflicts
                    if (!isDragging) {{
                        progress.style.width = percent + '%';
                        handle.style.left = percent + '%';
                        timeDisplay.innerHTML = 
                            formatTime(video.currentTime) + ' / ' + formatTime(video.duration);
                    }}
                    
                    // Update play button state
                    if (video.paused) {{
                        playIcon.textContent = '▶️';
                        playText.textContent = 'Play';
                    }} else {{
                        playIcon.textContent = '⏸️';
                        playText.textContent = 'Pause';
                    }}
                    
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
            
            // Render session markers on timeline
            function renderSessionMarkers() {{
                markers.innerHTML = '';
                elicitationSessions.forEach(session => {{
                    const startPercent = (session.startTime / video.duration) * 100;
                    const endPercent = (session.endTime / video.duration) * 100;
                    const width = Math.max(0.5, endPercent - startPercent);
                    
                    const marker = document.createElement('div');
                    marker.className = 'session-marker';
                    marker.style.left = startPercent + '%';
                    marker.style.width = width + '%';
                    marker.title = `Session ${{session.id}}: ${{formatTime(session.startTime)}} - ${{formatTime(session.endTime)}}`;
                    markers.appendChild(marker);
                }});
            }}
              // Event listeners
            playBtn.addEventListener('click', () => {{
                if (video.paused) {{
                    video.play();
                }} else {{
                    video.pause();
                }}
            }});
            
            // Timeline clicking and scrubbing
            timeline.addEventListener('click', (e) => {{
                if (!isLoaded) return;
                updateVideoTime(e);
            }});
            
            timeline.addEventListener('mousedown', (e) => {{
                if (!isLoaded) return;
                isDragging = true;
                updateVideoTime(e);
                e.preventDefault();
            }});
            
            document.addEventListener('mousemove', (e) => {{
                if (isDragging) {{
                    updateVideoTime(e);
                }}
            }});
              document.addEventListener('mouseup', () => {{
                if (isDragging) {{
                    isDragging = false;
                    // Force timeline update after dragging stops
                    updateTimeline();
                }}
            }});
              function updateVideoTime(e) {{
                const rect = timeline.getBoundingClientRect();
                const percent = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
                const newTime = percent * video.duration;
                
                // Set video time immediately
                video.currentTime = newTime;
                
                // Update visual timeline immediately during interaction
                const percentDisplay = percent * 100;
                progress.style.width = percentDisplay + '%';
                handle.style.left = percentDisplay + '%';
                timeDisplay.innerHTML = formatTime(newTime) + ' / ' + formatTime(video.duration);
                
                // Send update to Streamlit immediately
                window.parent.postMessage({{
                    type: 'video_time_update',
                    currentTime: newTime,
                    duration: video.duration,
                    isPlaying: !video.paused,
                    isRecording: isRecording
                }}, '*');
            }}
            
            // Recording functionality
            recordBtn.addEventListener('click', () => {{
                if (!isLoaded) {{
                    alert('Please wait for the video to load completely.');
                    return;
                }}
                
                if (!isRecording) {{
                    // Start recording
                    isRecording = true;
                    recordingStartTime = video.currentTime;
                    recordBtn.innerHTML = '⏹️ Stop Recording';
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
                    renderSessionMarkers();
                    
                    // Send session to Streamlit
                    window.parent.postMessage({{
                        type: 'elicitation_session',
                        session: session
                    }}, '*');
                    
                    // Reset recording state
                    isRecording = false;
                    recordingStartTime = null;
                    recordBtn.innerHTML = '🔴 Record';
                    recordBtn.className = 'btn btn-danger';
                    recordingIndicator.classList.remove('active');
                    
                    video.pause();
                }}
            }});
            
            // Display elicitation session
            function displaySession(session) {{
                // Clear "no sessions" message if it's the first session
                if (elicitationSessions.length === 1) {{
                    sessionsList.innerHTML = '';
                }}
                
                const sessionDiv = document.createElement('div');
                sessionDiv.className = 'session';
                sessionDiv.innerHTML = `
                    <div class="session-title">Session ${{session.id}}</div>
                    <div class="session-details">
                        <strong>Time Range:</strong> ${{formatTime(session.startTime)}} → ${{formatTime(session.endTime)}}<br>
                        <strong>Duration:</strong> ${{formatTime(session.duration)}}<br>
                        <strong>Captured:</strong> ${{session.timestamp}}
                    </div>
                `;
                
                // Add click to seek functionality
                sessionDiv.addEventListener('click', () => {{
                    video.currentTime = session.startTime;
                    updateTimeline();
                }});
                sessionDiv.style.cursor = 'pointer';
                sessionDiv.title = 'Click to jump to this session';
                
                sessionsList.appendChild(sessionDiv);
            }}
            
            // Listen for messages from Streamlit
            window.addEventListener('message', (event) => {{
                if (event.data.type === 'seek_video') {{
                    video.currentTime = event.data.time;
                    updateTimeline();
                }}
            }});
            
            // Video event listeners
            video.addEventListener('loadstart', () => {{
                timeDisplay.innerHTML = '<span class="loading-spinner"></span>Loading...';
            }});
            
            video.addEventListener('loadedmetadata', () => {{
                isLoaded = true;
                updateTimeline();
                console.log('Video metadata loaded:', {{
                    duration: video.duration,
                    videoWidth: video.videoWidth,
                    videoHeight: video.videoHeight
                }});
                
                // Send initial state to Streamlit
                window.parent.postMessage({{
                    type: 'video_loaded',
                    duration: video.duration,
                    width: video.videoWidth,
                    height: video.videoHeight
                }}, '*');
            }});
            
            video.addEventListener('canplay', () => {{
                console.log('Video can start playing');
            }});
            
            video.addEventListener('timeupdate', updateTimeline);
            video.addEventListener('play', updateTimeline);
            video.addEventListener('pause', updateTimeline);
            video.addEventListener('seeking', updateTimeline);
            video.addEventListener('seeked', updateTimeline);
            
            // Error handling
            video.addEventListener('error', (e) => {{
                console.error('Video error:', e);
                timeDisplay.innerHTML = '❌ Error loading video';
            }});
            
            // Initialize
            updateTimeline();
        </script>
    </body>
    </html>
    """
