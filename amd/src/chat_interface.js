/**
 * CraftPilot Chat Interface — site-wide widget (local_craftpilot)
 *
 * Adapted from mod_craftpilot: removed cmId / instanceId, widget is
 * injected on every authenticated Moodle page via page_init / before_footer.
 *
 * @module     local_craftpilot/chat_interface
 * @copyright  2026
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

import * as Ajax from 'core/ajax';
import * as MarkedLib from 'local_craftpilot/marked';
import DOMPurify from 'local_craftpilot/dompurify';

/* ============================================================
   STATE
   ============================================================ */
const state = {
    courseId:      null,   // retrieval hint; 0 = no course context
    chatProxyUrl:  null,   // set from PHP via init()
    userId:        0,      // set from PHP via init() — $USER->id
    currentConvId:    null,
    currentConvTitle: 'CraftPilot',
    conversations: [],
    sidebarOpen:   false,
    chatOpen:      false,
    streaming:     false,
    streamAbort:   null,    // AbortController for the active fetch; null when idle
    sources:       [],
    selectedDomain: null,
    followUpQuestions: { A: null, B: null, C: null },
    isNewConv:     false,  // true after createConversation, reset on first send
};

/* Per-conversation state — preserves sources across conv switches */
const convStates = {};
const getConvState = (id) => {
    if (!id) return { sources: [] };
    const key = String(id);
    if (!convStates[key]) convStates[key] = { sources: [] };
    return convStates[key];
};

/* DOM element references — populated in initDOM() */
const dom = {};

/* ============================================================
   PUBLIC INIT (AMD entry point)
   ============================================================ */
export const init = (moduleCourseId, chatProxyUrl, userId) => {
    state.courseId     = moduleCourseId || 0;
    state.chatProxyUrl = chatProxyUrl || '/local/craftpilot/chat_proxy.php';
    state.userId       = userId || 0;

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setup);
    } else {
        setup();
    }
};

/* ============================================================
   SETUP
   ============================================================ */
const setup = () => {
    if (!initDOM()) return;
    bindEvents();
    loadConversations();
};

const initDOM = () => {
    const map = {
        toggle:        'cp-toggle',
        wrapper:       'cp-wrapper',
        chatBody:      'cp-chat-body',
        sidebar:       'cp-sidebar',
        sidebarToggle: 'cp-sidebar-toggle',
        newConv:       'cp-new-conv',
        convList:      'cp-conv-list',
        panel:         'cp-panel',
        convTitle:     'cp-conv-title',
        closeBtn:      'cp-close',
        messages:      'cp-messages',
        sources:       'cp-sources',
        sourcesHeader: 'cp-sources-header',
        sourcesCount:  'cp-sources-count',
        sourcesScroll: 'cp-sources-scroll',
        domainBar:     'cp-domain-bar',
        input:         'cp-input',
        sendBtn:       'cp-send',
    };

    let ok = true;
    const required = new Set(['toggle', 'wrapper', 'messages', 'input', 'sendBtn']);

    Object.entries(map).forEach(([key, id]) => {
        dom[key] = document.getElementById(id);
        if (!dom[key] && required.has(key)) {
            console.error('CraftPilot: missing required element #' + id);
            ok = false;
        }
    });
    return ok;
};

/* ============================================================
   EVENT BINDING
   ============================================================ */
const bindEvents = () => {
    dom.toggle.addEventListener('click', toggleChat);
    dom.toggle.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleChat(); }
    });

    dom.closeBtn?.addEventListener('click', () => setChat(false));
    dom.sidebarToggle?.addEventListener('click', toggleSidebar);
    dom.newConv?.addEventListener('click', startNewConversation);

    dom.sendBtn.addEventListener('click', sendMessage);
    dom.input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });
    dom.input.addEventListener('input', autoResize);

    dom.sourcesHeader?.addEventListener('click', toggleSources);

    dom.domainBar?.addEventListener('click', (e) => {
        const btn = e.target.closest('.cp-domain-btn');
        if (!btn) return;
        const domain = btn.dataset.domain;
        if (state.selectedDomain === domain) {
            state.selectedDomain = null;
            btn.classList.remove('cp-domain-btn--active');
        } else {
            state.selectedDomain = domain;
            dom.domainBar.querySelectorAll('.cp-domain-btn').forEach(b => b.classList.remove('cp-domain-btn--active'));
            btn.classList.add('cp-domain-btn--active');
        }
    });

    // Abort any in-flight stream when the user navigates away or refreshes,
    // so the browser is not blocked waiting for the fetch to settle.
    window.addEventListener('beforeunload', () => {
        if (state.streamAbort) state.streamAbort.abort();
    });
};

/* ============================================================
   CHAT OPEN / CLOSE
   ============================================================ */
const setChat = (open) => {
    state.chatOpen = open;
    dom.wrapper.classList.toggle('cp-wrapper--open', open);
    dom.chatBody?.setAttribute('aria-hidden', String(!open));
    dom.toggle.setAttribute('aria-expanded', String(open));
    dom.toggle.classList.toggle('cp-toggle-pill--active', open);
    if (open) setTimeout(() => dom.input?.focus(), 50);
};

const toggleChat = () => setChat(!state.chatOpen);

/* ============================================================
   SIDEBAR
   ============================================================ */
const toggleSidebar = () => {
    state.sidebarOpen = !state.sidebarOpen;
    dom.sidebar?.classList.toggle('cp-sidebar--open', state.sidebarOpen);
    dom.wrapper?.classList.toggle('cp-wrapper--sidebar-open', state.sidebarOpen);
};

/* ============================================================
   SOURCES SECTION
   ============================================================ */
const toggleSources = () => {
    const open = dom.sources.classList.toggle('cp-sources--open');
    dom.sourcesHeader?.setAttribute('aria-expanded', String(open));
};

const setSources = (items) => {
    state.sources = items;
    if (!dom.sources || !dom.sourcesScroll) return;

    if (!items.length) {
        dom.sources.classList.remove('cp-sources--visible', 'cp-sources--open');
        dom.sourcesScroll.innerHTML = '';
        return;
    }

    dom.sources.classList.add('cp-sources--visible', 'cp-sources--open');
    dom.sourcesHeader?.setAttribute('aria-expanded', 'true');
    if (dom.sourcesCount) dom.sourcesCount.textContent = items.length;

    dom.sourcesScroll.innerHTML = '';
    items.forEach(item => dom.sourcesScroll.appendChild(buildSourceCard(item)));
};

const addSource = (item, ownerConvId) => {
    const targetConvId = ownerConvId || state.currentConvId;
    const convSrcs = getConvState(targetConvId).sources;
    if (convSrcs.some(s => s.id === item.id)) return;
    convSrcs.push(item);
    // Only update the visible DOM if this source belongs to the active conversation
    if (String(targetConvId) === String(state.currentConvId)) {
        state.sources = convSrcs;
        setSources(state.sources);
    }
};

const clearSources = () => setSources([]);

/* ============================================================
   SOURCE CARDS
   ============================================================ */
const buildSourceCard = (item) => {
    const card = document.createElement('div');
    card.className = 'cp-source-card cp-card--' + item.type;
    card.setAttribute('role', 'listitem');
    card.setAttribute('tabindex', '0');
    card.setAttribute('aria-label', (item.filename || item.type) + ' — click to open');

    if (item.type === 'video') {
        card.innerHTML =
            '<div class="cp-card-thumb">' +
                '<div class="cp-card-play">' +
                    '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">' +
                        '<path d="M8 5v14l11-7z"/>' +
                    '</svg>' +
                '</div>' +
            '</div>' +
            '<div class="cp-card-footer">' +
                '<span class="cp-card-badge">Video</span>' +
                '<span class="cp-card-name">' + escapeHtml(item.filename || 'Video clip') + '</span>' +
            '</div>';
        card.addEventListener('click', () => openVideoModal(item));

    } else if (item.type === 'bvh') {
        card.innerHTML =
            '<div class="cp-card-thumb">' +
                '<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">' +
                    '<circle cx="12" cy="3.5" r="1.5"/>' +
                    '<line x1="12" y1="5"  x2="12" y2="11"/>' +
                    '<line x1="12" y1="8"  x2="8"  y2="6"/>' +
                    '<line x1="12" y1="8"  x2="16" y2="6"/>' +
                    '<line x1="12" y1="11" x2="10" y2="17"/>' +
                    '<line x1="12" y1="11" x2="14" y2="17"/>' +
                    '<line x1="10" y1="17" x2="9.5" y2="21"/>' +
                    '<line x1="14" y1="17" x2="14.5" y2="21"/>' +
                '</svg>' +
            '</div>' +
            '<div class="cp-card-footer">' +
                '<span class="cp-card-badge">Motion</span>' +
                '<span class="cp-card-name">' + escapeHtml(item.filename || 'Motion data') + '</span>' +
            '</div>';
        card.addEventListener('click', () => openBVHModal(item));

    } else if (item.type === 'text') {
        const preview = escapeHtml((item.content || '').substring(0, 200));
        card.innerHTML =
            '<div class="cp-card-thumb">' +
                '<div class="cp-text-preview-wrap">' +
                    '<div class="cp-text-blur-top"></div>' +
                    '<span class="cp-text-highlight">' + preview + '</span>' +
                    '<div class="cp-text-blur-bottom"></div>' +
                '</div>' +
            '</div>' +
            '<div class="cp-card-footer">' +
                '<span class="cp-card-badge">Text</span>' +
                '<span class="cp-card-name">' + escapeHtml(item.source || 'Document') + '</span>' +
            '</div>';
        card.addEventListener('click', () => openTextModal(item));
    }

    card.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); card.click(); } });
    return card;
};

/* ============================================================
   VIDEO MODAL
   ============================================================ */
const openVideoModal = (item) => {
    const videoUrl = '/local/craftpilot/video_proxy.php?id=' + encodeURIComponent(item.id || '');
    const startMs  = item.start_time != null ? item.start_time : null;
    const endMs    = item.end_time   != null ? item.end_time   : null;

    const {overlay, modal, closeModal} = createModal();

    modal.innerHTML =
        '<div class="cp-modal-header">' +
            '<h2 class="cp-modal-title">' + escapeHtml(item.filename || item.project_name || 'Video') + '</h2>' +
            '<button class="cp-icon-btn cp-modal-close" aria-label="Close">' +
                svgClose() +
            '</button>' +
        '</div>' +
        '<div class="cp-modal-body">' +
            '<video class="cp-video-player" controls preload="metadata">' +
                '<source src="' + videoUrl + '" type="video/mp4">' +
                'Your browser does not support the video element.' +
            '</video>' +
            (startMs !== null
                ? '<p style="margin:12px 0 0;font-size:13px;color:#767676;font-family:var(--cp-font-body)">' +
                      'Segment: ' + formatTime(startMs) + ' — ' + formatTime(endMs || 0) +
                  '</p>'
                : '') +
        '</div>';

    modal.querySelector('.cp-modal-close').addEventListener('click', closeModal);

    const video = modal.querySelector('video');
    if (startMs !== null) {
        video.addEventListener('loadedmetadata', () => { video.currentTime = startMs; });
    }

    document.body.appendChild(overlay);
    requestAnimationFrame(() => overlay.classList.add('cp-modal-overlay--visible'));
};

/* ============================================================
   BVH MODAL  (Three.js skeleton viewer)
   ============================================================ */
const openBVHModal = (item) => {
    const rawUrl = item.bvh_url || item.url || '';
    const backendBase = state.chatProxyUrl ? state.chatProxyUrl.replace('/local/craftpilot/chat_proxy.php', '') : '';
    const bvhUrl = rawUrl.startsWith('http') ? rawUrl : backendBase + rawUrl;
    const canvasId = 'cp-bvh-canvas-' + Date.now();

    const {overlay, modal, closeModal} = createModal();

    modal.innerHTML =
        '<div class="cp-modal-header">' +
            '<h2 class="cp-modal-title">' + escapeHtml(item.filename || 'Motion Capture') + '</h2>' +
            '<button class="cp-icon-btn cp-modal-close" aria-label="Close">' +
                svgClose() +
            '</button>' +
        '</div>' +
        '<div class="cp-modal-body">' +
            '<div class="cp-bvh-canvas-wrap" id="' + canvasId + '">' +
                '<div class="cp-bvh-loading">' +
                    '<div class="cp-spinner"></div>' +
                    '<span>Loading motion data\u2026</span>' +
                '</div>' +
            '</div>' +
            '<div class="cp-bvh-controls">' +
                '<button class="cp-bvh-play" id="cp-bvh-play-btn" aria-label="Play / Pause">' +
                    svgPlay() +
                '</button>' +
                '<input type="range" class="cp-bvh-scrubber" min="0" max="1000" value="0" step="1" aria-label="Playback position">' +
                '<select class="cp-bvh-speed" aria-label="Playback speed">' +
                    '<option value="0.25">0.25\u00d7</option>' +
                    '<option value="0.5">0.5\u00d7</option>' +
                    '<option value="1" selected>1\u00d7</option>' +
                    '<option value="2">2\u00d7</option>' +
                '</select>' +
            '</div>' +
        '</div>';

    const doClose = () => {
        const v = window._cpBvhViewer;
        if (v) { v.destroy(); window._cpBvhViewer = null; }
        closeModal();
    };

    modal.querySelector('.cp-modal-close').addEventListener('click', doClose);

    document.body.appendChild(overlay);
    requestAnimationFrame(() => overlay.classList.add('cp-modal-overlay--visible'));

    const container = document.getElementById(canvasId);
    const playBtn   = modal.querySelector('#cp-bvh-play-btn');
    const scrubber  = modal.querySelector('.cp-bvh-scrubber');
    const speedSel  = modal.querySelector('.cp-bvh-speed');

    loadThreeJS()
        .then(() => {
            const viewer = new BVHViewer(container, bvhUrl);
            window._cpBvhViewer = viewer;
            return viewer.init();
        })
        .then(() => {
            const viewer = window._cpBvhViewer;

            playBtn.addEventListener('click', () => {
                viewer.togglePlay();
                playBtn.innerHTML = viewer.playing ? svgPause() : svgPlay();
            });

            viewer.onProgress = (p) => {
                scrubber.value = Math.round(p * 1000);
            };

            scrubber.addEventListener('input', (e) => {
                viewer.seekTo(parseInt(e.target.value, 10) / 1000);
            });

            speedSel.addEventListener('change', (e) => {
                viewer.speed = parseFloat(e.target.value);
            });
        })
        .catch((err) => {
            container.innerHTML =
                '<div class="cp-bvh-loading">' +
                    '<p style="color:#dc2626;font-family:var(--cp-font-body);font-size:13px">' +
                        'Could not load BVH: ' + escapeHtml(err.message) +
                    '</p>' +
                '</div>';
        });
};

/* ============================================================
   TEXT MODAL  — redirects to the Moodle course module page
   when module info is available; falls back to content preview
   ============================================================ */
const openTextModal = (item) => {
    if (item.module_id && item.module_type) {
        let moduleUrl;
        if (item.module_type === 'label') {
            moduleUrl = '/course/view.php?id=' + encodeURIComponent(item.course_id)
                      + '#module-' + encodeURIComponent(item.module_id);
        } else {
            moduleUrl = '/mod/' + encodeURIComponent(item.module_type)
                      + '/view.php?id=' + encodeURIComponent(item.module_id);
        }
        window.open(moduleUrl, '_blank', 'noopener');
        return;
    }

    /* Fallback: show content in a modal when no module link available */
    const {overlay, modal, closeModal} = createModal();

    modal.innerHTML =
        '<div class="cp-modal-header">' +
            '<h2 class="cp-modal-title">' + escapeHtml(item.source || 'Document') + '</h2>' +
            '<button class="cp-icon-btn cp-modal-close" aria-label="Close">' +
                svgClose() +
            '</button>' +
        '</div>' +
        '<div class="cp-text-modal-body">' +
            '<p>' + escapeHtml(item.content || '') + '</p>' +
        '</div>';

    modal.querySelector('.cp-modal-close').addEventListener('click', closeModal);
    document.body.appendChild(overlay);
    requestAnimationFrame(() => overlay.classList.add('cp-modal-overlay--visible'));
};

/* ============================================================
   MODAL UTILITIES
   ============================================================ */
const createModal = () => {
    const overlay = document.createElement('div');
    overlay.className = 'cp-modal-overlay';

    const modal = document.createElement('div');
    modal.className = 'cp-modal';
    overlay.appendChild(modal);

    const closeModal = () => {
        overlay.classList.remove('cp-modal-overlay--visible');
        setTimeout(() => { if (overlay.parentNode) overlay.parentNode.removeChild(overlay); }, 320);
    };

    overlay.addEventListener('click', (e) => { if (e.target === overlay) closeModal(); });

    const onEsc = (e) => {
        if (e.key === 'Escape') { closeModal(); document.removeEventListener('keydown', onEsc); }
    };
    document.addEventListener('keydown', onEsc);

    return {overlay, modal, closeModal};
};

/* ============================================================
   LAZY-LOAD THREE.JS
   ============================================================ */
const loadThreeJS = () => new Promise((resolve, reject) => {
    if (window.THREE) { resolve(); return; }
    const s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js';
    s.onload  = resolve;
    s.onerror = () => reject(new Error('Failed to load Three.js'));
    document.head.appendChild(s);
});

/* ============================================================
   BVH VIEWER  (Three.js WebGL skeleton animator)
   ============================================================ */
class BVHViewer {
    constructor(container, url) {
        this.container   = container;
        this.url         = url;
        this.playing     = false;
        this.speed       = 1;
        this.frameIndex  = 0;
        this.frames      = [];
        this.frameTime   = 1 / 30;
        this.elapsed     = 0;
        this.lastTs      = 0;
        this.animFrame   = null;
        this.onProgress  = null;

        this._scene      = null;
        this._camera     = null;
        this._renderer   = null;
        this._root       = null;
        this._bonePairs  = [];
        this._bonePos    = null;
        this._boneLines  = null;
        this._spheres    = [];
    }

    async init() {
        const THREE = window.THREE;
        const w = Math.max(this.container.clientWidth  || 520, 100);
        const h = 320;

        this._scene = new THREE.Scene();
        this._scene.background = new THREE.Color(0xfafafa);

        this._camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 50000);

        this._renderer = new THREE.WebGLRenderer({ antialias: true });
        this._renderer.setSize(w, h);
        this._renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

        this.container.innerHTML = '';
        this.container.appendChild(this._renderer.domElement);

        const grid = new THREE.GridHelper(1200, 30, 0xe8e8e8, 0xededed);
        this._scene.add(grid);

        this._scene.add(new THREE.AmbientLight(0xffffff, 1));

        const res = await fetch(this.url);
        if (!res.ok) throw new Error('HTTP ' + res.status + ' fetching BVH');
        const text = await res.text();
        const parsed = parseBVH(text);

        this._root     = parsed.root;
        this.frames    = parsed.frames;
        this.frameTime = parsed.frameTime;

        if (this.frames.length > 0) {
            applyFrame(this._root, this.frames[0]);
        }

        this._buildGeometry();
        this._updateGeometry();
        this._autofitCamera();
        this._animate(0);
    }

    _buildGeometry() {
        const THREE = window.THREE;

        const collect = (joint) => {
            joint.children.forEach(child => {
                this._bonePairs.push([joint, child]);
                if (!child.isEndSite) collect(child);
            });
        };
        collect(this._root);

        const n = this._bonePairs.length;
        this._bonePos = new Float32Array(n * 6);

        const geo = new THREE.BufferGeometry();
        const attr = new THREE.BufferAttribute(this._bonePos, 3);
        attr.setUsage(THREE.DynamicDrawUsage);
        geo.setAttribute('position', attr);
        geo.setDrawRange(0, n * 2);

        const mat = new THREE.LineBasicMaterial({ color: 0x0f6cbf, linewidth: 2 });
        this._boneLines = new THREE.LineSegments(geo, mat);
        this._scene.add(this._boneLines);

        const sGeo = new THREE.SphereGeometry(3, 6, 6);
        const sMat = new THREE.MeshBasicMaterial({ color: 0x000000 });

        const addSpheres = (joint) => {
            if (!joint.isEndSite) {
                const mesh = new THREE.Mesh(sGeo, sMat);
                this._scene.add(mesh);
                this._spheres.push({ joint, mesh });
                joint.children.forEach(c => addSpheres(c));
            }
        };
        addSpheres(this._root);
    }

    _updateGeometry() {
        let i = 0;
        this._bonePairs.forEach(([parent, child]) => {
            const pp = parent._worldPos;
            const cp = child._worldPos || pp;
            this._bonePos[i++] = pp.x;
            this._bonePos[i++] = pp.y;
            this._bonePos[i++] = pp.z;
            this._bonePos[i++] = cp.x;
            this._bonePos[i++] = cp.y;
            this._bonePos[i++] = cp.z;
        });
        this._boneLines.geometry.attributes.position.needsUpdate = true;

        this._spheres.forEach(({ joint, mesh }) => {
            if (joint._worldPos) mesh.position.copy(joint._worldPos);
        });
    }

    _autofitCamera() {
        const THREE = window.THREE;
        const box = new THREE.Box3();
        this._spheres.forEach(({ joint }) => {
            if (joint._worldPos) box.expandByPoint(joint._worldPos);
        });

        const center = new THREE.Vector3();
        const size   = new THREE.Vector3();
        box.getCenter(center);
        box.getSize(size);

        const maxDim = Math.max(size.x, size.y, size.z, 1);
        const dist   = maxDim * 2.0;

        this._camera.position.set(center.x, center.y + size.y * 0.05, center.z + dist);
        this._camera.lookAt(center);
        this._camera.far = dist * 12;
        this._camera.updateProjectionMatrix();
    }

    _animate(ts) {
        this.animFrame = requestAnimationFrame((t) => this._animate(t));

        if (this.playing && this.frames.length > 0) {
            const delta = (ts - this.lastTs) / 1000;
            this.elapsed += delta * this.speed;

            const total = this.frames.length * this.frameTime;
            if (this.elapsed >= total) this.elapsed = this.elapsed % total;

            this.frameIndex = Math.min(
                Math.floor(this.elapsed / this.frameTime),
                this.frames.length - 1
            );

            applyFrame(this._root, this.frames[this.frameIndex]);
            this._updateGeometry();

            if (this.onProgress) this.onProgress(this.elapsed / total);
        }

        this.lastTs = ts;
        this._renderer.render(this._scene, this._camera);
    }

    togglePlay() { this.playing = !this.playing; }

    seekTo(progress) {
        const total = this.frames.length * this.frameTime;
        this.elapsed    = progress * total;
        this.frameIndex = Math.min(
            Math.floor(this.elapsed / this.frameTime),
            this.frames.length - 1
        );
        applyFrame(this._root, this.frames[this.frameIndex]);
        this._updateGeometry();
        if (this._renderer) this._renderer.render(this._scene, this._camera);
    }

    destroy() {
        if (this.animFrame) {
            cancelAnimationFrame(this.animFrame);
            this.animFrame = null;
        }
        if (this._renderer) {
            this._renderer.dispose();
            this._renderer = null;
        }
        if (this._boneLines) {
            this._boneLines.geometry.dispose();
            this._boneLines.material.dispose();
        }
    }
}

/* ============================================================
   BVH PARSER
   ============================================================ */
const parseBVH = (text) => {
    const lines  = text.split(/\r?\n/);
    let cursor   = 0;

    const read = () => {
        while (cursor < lines.length) {
            const l = lines[cursor++].trim();
            if (l) return l;
        }
        return null;
    };

    let line;
    while ((line = read()) && line !== 'HIERARCHY') { /* skip */ }

    const allJoints = [];

    const parseJoint = (parent) => {
        const decl  = read();
        const parts = decl.split(/\s+/);
        const isEnd = parts[0] === 'End';
        const name  = isEnd ? ((parent ? parent.name : '') + '_end') : parts[1];

        const joint = {
            name,
            parent,
            offset:    [0, 0, 0],
            channels:  [],
            children:  [],
            isEndSite: isEnd,
            _worldPos: null,
        };

        if (!isEnd) allJoints.push(joint);

        read(); /* opening { */

        let l;
        while ((l = read()) !== '}') {
            const p = l.split(/\s+/);
            if (p[0] === 'OFFSET') {
                joint.offset = [parseFloat(p[1]), parseFloat(p[2]), parseFloat(p[3])];
            } else if (p[0] === 'CHANNELS') {
                const n = parseInt(p[1], 10);
                for (let i = 0; i < n; i++) joint.channels.push(p[2 + i]);
            } else if (p[0] === 'JOINT') {
                cursor--;
                const child = parseJoint(joint);
                joint.children.push(child);
            } else if (p[0] === 'End') {
                cursor--;
                const endSite = parseJoint(joint);
                joint.children.push(endSite);
            }
        }
        return joint;
    };

    const root = parseJoint(null);

    while ((line = read()) && line !== 'MOTION') { /* skip */ }

    const framesLine = read();
    const numFrames  = parseInt((framesLine || '').replace(/[^0-9]/g, ''), 10) || 0;

    const ftLine    = read();
    const ftParts   = (ftLine || '').split(':');
    const frameTime = parseFloat(ftParts[1] != null ? ftParts[1] : ftParts[0]) || (1 / 30);

    const frames = [];
    for (let i = 0; i < numFrames; i++) {
        const fl = read();
        if (!fl) break;
        frames.push(fl.trim().split(/\s+/).map(parseFloat));
    }

    return { root, joints: allJoints, frames, frameTime };
};

/* ============================================================
   BVH FORWARD KINEMATICS
   ============================================================ */
const DEG2RAD = Math.PI / 180;

const applyFrame = (root, frame) => {
    if (!frame || !window.THREE) return;
    const THREE = window.THREE;
    const ref   = { idx: 0 };

    const traverse = (joint, parentMat) => {
        let tx = joint.offset[0];
        let ty = joint.offset[1];
        let tz = joint.offset[2];
        let rx = 0, ry = 0, rz = 0;
        const rotAxes = [];

        if (!joint.isEndSite) {
            joint.channels.forEach((ch) => {
                const val = (frame[ref.idx] !== undefined ? frame[ref.idx] : 0);
                ref.idx++;
                switch (ch) {
                    case 'Xposition': tx += val; break;
                    case 'Yposition': ty += val; break;
                    case 'Zposition': tz += val; break;
                    case 'Xrotation': rx = val * DEG2RAD; rotAxes.push('X'); break;
                    case 'Yrotation': ry = val * DEG2RAD; rotAxes.push('Y'); break;
                    case 'Zrotation': rz = val * DEG2RAD; rotAxes.push('Z'); break;
                }
            });
        }

        const order     = rotAxes.join('') || 'ZXY';
        const T         = new THREE.Matrix4().makeTranslation(tx, ty, tz);
        const R         = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(rx, ry, rz, order));
        const localMat  = new THREE.Matrix4().multiplyMatrices(T, R);
        const worldMat  = parentMat
            ? new THREE.Matrix4().multiplyMatrices(parentMat, localMat)
            : localMat.clone();

        joint._worldPos = new THREE.Vector3().setFromMatrixPosition(worldMat);
        joint.children.forEach(c => traverse(c, worldMat));
    };

    traverse(root, null);
};

/* ============================================================
   CONVERSATIONS
   ============================================================ */
const loadConversations = () => {
    Ajax.call([{
        methodname: 'local_craftpilot_manage_conversations',
        args: { action: 'list', courseid: state.courseId || 0 },
    }])[0].then((resp) => {
        const convs = resp.conversations || resp.data || [];
        if (resp.success && convs.length > 0) {
            state.conversations = convs;
            const first = convs[0];
            selectConversation(first.conversation_id || first.id, first.title || 'Chat');
        } else {
            createConversation(generateUUID());
        }
    }).catch((err) => {
        console.error('CraftPilot:', err);
        showError('Failed to initialise chat. Please refresh the page.');
    });
};

const createConversation = (id) => {
    // Clear UI immediately (sync) — prevents race that shows old conv content
    state.currentConvId    = id;
    state.currentConvTitle = 'New conversation';
    state.isNewConv        = true;
    state.conversations.unshift({ conversation_id: id, title: 'New conversation', created_time: Math.floor(Date.now() / 1000) });
    dom.messages.innerHTML = '';
    getConvState(id).sources = [];
    clearSources();
    renderConversations(state.conversations, id);
    updatePanelTitle(state.currentConvTitle);
    showReady();
    // Persist to backend — fire-and-forget
    Ajax.call([{
        methodname: 'local_craftpilot_manage_conversations',
        args: {
            action:          'create',
            conversation_id: id,
            title:           'New conversation',
            courseid:        state.courseId || 0,
            metadata:        JSON.stringify({ created: new Date().toISOString() }),
        },
    }])[0].catch(err => console.error('CraftPilot:', err));
};

const startNewConversation = () => createConversation(generateUUID());

const selectConversation = (id, title) => {
    // Save sources for the conversation we're leaving
    if (state.currentConvId) {
        getConvState(state.currentConvId).sources = [...state.sources];
    }
    state.currentConvId    = id;
    state.currentConvTitle = title;
    updatePanelTitle(title);
    renderConversations(state.conversations, id);
    dom.messages.innerHTML = '';
    // Restore saved sources (or clear if none for this conv)
    setSources(getConvState(id).sources);
    loadMessages(id);
};

const deleteConversation = (id) => {
    if (!window.confirm('Delete this conversation and all its messages?')) return;
    Ajax.call([{
        methodname: 'local_craftpilot_manage_conversations',
        args: { action: 'delete', conversation_id: id },
    }])[0].then((resp) => {
        if (!resp.success) return;
        delete convStates[String(id)];
        state.conversations = state.conversations.filter(
            c => String(c.conversation_id || c.id) !== String(id)
        );
        if (String(state.currentConvId) === String(id)) {
            dom.messages.innerHTML = '';
            clearSources();

            if (state.conversations.length > 0) {
                const next = state.conversations[0];
                selectConversation(next.conversation_id || next.id, next.title || 'Chat');
            } else {
                createConversation(generateUUID());
            }
        } else {
            renderConversations(state.conversations, state.currentConvId);
        }
    }).catch(err => console.error('CraftPilot:', err));
};

/* ============================================================
   CONVERSATION LIST RENDER
   ============================================================ */
const renderConversations = (convs, activeId) => {
    if (!dom.convList) return;
    dom.convList.innerHTML = '';

    convs.forEach((conv) => {
        const id   = String(conv.conversation_id || conv.id);
        const item = document.createElement('div');
        item.className = 'cp-conv-item' + (id === String(activeId) ? ' cp-conv-item--active' : '');
        item.setAttribute('role', 'listitem');
        item.dataset.id = id;

        const ts   = (conv.created_time || 0);
        const ms   = ts > 1e10 ? ts : ts * 1000;
        const date = new Date(ms).toLocaleDateString();

        item.innerHTML =
            '<div class="cp-conv-info">' +
                '<span class="cp-conv-title">' + escapeHtml(conv.title || 'Chat') + '</span>' +
                '<span class="cp-conv-date">' + date + '</span>' +
            '</div>' +
            '<button class="cp-conv-delete" aria-label="Delete conversation" title="Delete">' +
                '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">' +
                    '<polyline points="3 6 5 6 21 6"/>' +
                    '<path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>' +
                '</svg>' +
            '</button>';

        item.addEventListener('click', (e) => {
            if (e.target.closest('.cp-conv-delete')) {
                e.stopPropagation();
                deleteConversation(id);
            } else {
                selectConversation(id, conv.title || 'Chat');
            }
        });

        dom.convList.appendChild(item);
    });
};

/* ============================================================
   MESSAGES
   ============================================================ */
const loadMessages = (convId) => {
    Ajax.call([{
        methodname: 'local_craftpilot_manage_messages',
        args: { action: 'load', conversation_id: convId },
    }])[0].then((resp) => {
        if (String(convId) !== String(state.currentConvId)) return; // stale response
        const msgs = resp.messages || resp.data || [];
        if (resp.success && msgs.length > 0) {
            dom.messages.innerHTML = '';
            msgs.forEach(m => appendMessage(
                (m.message_type === 'user' || m.role === 'user') ? 'user' : 'ai',
                m.content,
                false
            ));
            scrollBottom();
            // Restore sources from the last AI message's metadata (only if none already in memory)
            if (getConvState(convId).sources.length === 0) {
                const lastAi = [...msgs].reverse().find(m => m.message_type === 'ai' || m.role === 'ai');
                if (lastAi && lastAi.metadata) {
                    try {
                        const meta = JSON.parse(lastAi.metadata);
                        if (Array.isArray(meta.sources) && meta.sources.length > 0) {
                            getConvState(convId).sources = meta.sources;
                            setSources(meta.sources);
                        }
                    } catch (_) { /* malformed metadata — skip */ }
                }
            }
        } else {
            showReady();
        }
    }).catch(err => console.error('CraftPilot:', err));
};

const appendMessage = (role, content, animate) => {
    hideEmpty();

    const wrapper = document.createElement('div');
    wrapper.className = 'cp-msg cp-msg--' + (role === 'user' ? 'user' : 'ai');
    if (animate === false) wrapper.style.animation = 'none';

    const bubble = document.createElement('div');
    bubble.className = 'cp-msg-bubble';

    if (role === 'user') {
        bubble.textContent = content;
    } else {
        const clean = stripThinkTags(content);
        bubble.innerHTML = renderMarkdown(clean);
        extractFollowUpQuestions(clean);
    }

    wrapper.appendChild(bubble);
    dom.messages.appendChild(wrapper);
    return wrapper;
};

const showTyping = () => {
    hideEmpty();
    const wrapper = document.createElement('div');
    wrapper.className = 'cp-msg cp-msg--ai';
    wrapper.id = 'cp-typing-indicator';
    wrapper.innerHTML =
        '<div class="cp-typing">' +
            '<div class="cp-typing-dot"></div>' +
            '<div class="cp-typing-dot"></div>' +
            '<div class="cp-typing-dot"></div>' +
            '<span class="cp-status-hint" aria-live="polite"></span>' +
        '</div>';
    dom.messages.appendChild(wrapper);
    scrollBottom();
    return wrapper;
};

const hideEmpty = () => {
    const el = document.getElementById('cp-empty-state');
    if (el) el.parentNode.removeChild(el);
};

const showReady = () => {
    if (document.getElementById('cp-empty-state')) return;
    const el = document.createElement('div');
    el.className = 'cp-empty-state';
    el.id = 'cp-empty-state';
    el.innerHTML =
        '<svg class="cp-empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">' +
            '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>' +
        '</svg>' +
        '<p class="cp-empty-title">Ready to assist</p>' +
        '<p class="cp-empty-sub">Ask a question about this content</p>';
    dom.messages.appendChild(el);
};

const showError = (msg) => {
    const el = appendMessage('ai', msg, true);
    el.querySelector('.cp-msg-bubble').style.color = '#dc2626';
};

/* ============================================================
   SEND MESSAGE + STREAM
   ============================================================ */
const sendMessage = () => {
    let text = dom.input.value.trim();
    if (!text || state.streaming) return;

    if (!state.currentConvId) {
        showError('Chat not ready yet — please wait a moment.');
        return;
    }

    if (/^[ABC]$/i.test(text)) {
        const letter = text.toUpperCase();
        const expanded = state.followUpQuestions[letter];
        if (expanded) text = expanded;
    }

    state.streaming      = true;
    dom.input.disabled   = true;
    dom.sendBtn.disabled = true;

    appendMessage('user', text, true);
    dom.input.value = '';
    autoResize();
    getConvState(state.currentConvId).sources = [];
    clearSources();
    scrollBottom();

    const isFirst = state.isNewConv;
    state.isNewConv = false;
    saveMessage(state.currentConvId, 'user', text);
    streamFromBackend(text, isFirst);
};

const streamFromBackend = (userMessage, isFirstMessage = false) => {
    const streamConvId = state.currentConvId; // capture now — don't read live inside stream
    const typingEl = showTyping();

    const controller = new AbortController();
    state.streamAbort = controller;

    const payload = {
        message: userMessage,
        conversation_thread_id: state.currentConvId,
        is_first_message: isFirstMessage,
        sesskey: (window.M && window.M.cfg) ? window.M.cfg.sesskey : '',
        user_id: state.userId,
    };
    if (state.selectedDomain) {
        payload.selected_domain = state.selectedDomain;
    }
    if (state.courseId) {
        payload.course_id = String(state.courseId);
    }
    Promise.resolve(fetch(state.chatProxyUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal,
    }))
        .then((res) => {
            if (String(streamConvId) !== String(state.currentConvId)) {
                res.body.cancel();
                finishStreaming();
                return;
            }
            if (!res.ok) throw new Error('Backend responded ' + res.status);

            // Typing indicator stays alive while status events cycle through.
            // msgEl/bubble are created lazily on the first token so the hint
            // remains visible during the full retrieval + reranking phase.
            let msgEl = null;
            let bubble = null;
            let fullResponse = '';

            const activateBubble = () => {
                if (msgEl) return bubble;
                if (typingEl.parentNode) typingEl.parentNode.removeChild(typingEl);
                msgEl = appendMessage('ai', '', true);
                bubble = msgEl.querySelector('.cp-msg-bubble');
                return bubble;
            };

            const reader  = res.body.getReader();
            const decoder = new TextDecoder();
            let buf = '';

            const read = async () => {
                const { done, value } = await reader.read();

                if (done) {
                    if (String(streamConvId) !== String(state.currentConvId)) {
                        finishStreaming();
                        return;
                    }
                    const b = activateBubble();
                    const cleanFinal = stripThinkTags(fullResponse);
                    b.innerHTML = renderMarkdown(cleanFinal);
                    extractFollowUpQuestions(cleanFinal);
                    const finishedSources = getConvState(streamConvId).sources;
                    saveMessage(streamConvId, 'ai', cleanFinal,
                        finishedSources.length ? JSON.stringify({sources: finishedSources}) : '');
                    finishStreaming();
                    return;
                }

                buf += decoder.decode(value, { stream: true });
                const lines = buf.split('\n');
                buf = lines.pop();

                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const ev = JSON.parse(line);

                        if (ev.event === 'conversation_title' && ev.data) {
                            const newTitle = ev.data;
                            state.currentConvTitle = newTitle;
                            updatePanelTitle(newTitle);
                            const conv = state.conversations.find(
                                c => String(c.conversation_id || c.id) === String(streamConvId)
                            );
                            if (conv) conv.title = newTitle;
                            renderConversations(state.conversations, streamConvId);
                            Ajax.call([{
                                methodname: 'local_craftpilot_manage_conversations',
                                args: { action: 'update', conversation_id: streamConvId, title: newTitle },
                            }])[0].catch(err => console.error('CraftPilot: title persist failed', err));

                        } else if (ev.event === 'status' && ev.data) {
                            const hintEl = typingEl.querySelector('.cp-status-hint');
                            if (hintEl) hintEl.textContent = ev.data;
                            // Yield one animation frame so the browser renders this
                            // hint before processing the next event in the same chunk.
                            await new Promise(resolve => requestAnimationFrame(resolve));

                        } else if (ev.event === 'video_metadata' && ev.data) {
                            const vm    = ev.data;
                            const isBVH = (vm.filename || '').toLowerCase().endsWith('.bvh');
                            addSource({
                                id:          vm.video_id || vm.bvh_id || vm.filename,
                                type:        isBVH ? 'bvh' : 'video',
                                filename:    vm.filename,
                                video_url:   isBVH ? null : vm.video_url,
                                bvh_url:     isBVH ? (vm.bvh_url || vm.video_url) : null,
                                start_time:  vm.start_time,
                                end_time:    vm.end_time,
                                duration:    vm.duration,
                                project_name: vm.project_name,
                            }, streamConvId);

                        } else if (ev.event === 'bvh_metadata' && ev.data) {
                            const bm = ev.data;
                            addSource({
                                id:          bm.bvh_id || bm.filename,
                                type:        'bvh',
                                filename:    bm.filename,
                                bvh_url:     bm.bvh_url || bm.url,
                                duration:    bm.duration,
                                frame_count: bm.frame_count,
                            }, streamConvId);

                        } else if (ev.event === 'token' && ev.data) {
                            fullResponse += ev.data;
                            const visible = stripThinkTags(fullResponse);
                            activateBubble().innerHTML = renderMarkdown(visible);
                            scrollBottom();

                        } else if (ev.event === 'message' && ev.content) {
                            ev.content.forEach((c) => { if (c.content) fullResponse += c.content; });
                            const visible = stripThinkTags(fullResponse);
                            activateBubble().innerHTML = renderMarkdown(visible);
                            scrollBottom();

                        } else if (ev.event === 'documents' && Array.isArray(ev.data)) {
                            ev.data.forEach((doc) => {
                                if (doc.type === 'video_annotation') return;
                                addSource({
                                    id:           doc.source,
                                    type:         'text',
                                    source:       doc.module_name || doc.source,
                                    content:      doc.page_content_preview,
                                    module_id:    doc.module_id,
                                    module_type:  doc.module_type,
                                    course_id:    doc.course_id,
                                    heading_path: doc.heading_path,
                                    section_name: doc.section_name,
                                }, streamConvId);
                            });

                        } else if (ev.content === '[DONE]') {
                            /* handled on stream close */

                        } else if (ev.event === 'error' || ev.type === 'error') {
                            if (msgEl && msgEl.parentNode) msgEl.parentNode.removeChild(msgEl);
                            if (typingEl.parentNode) typingEl.parentNode.removeChild(typingEl);
                            showError(ev.message || 'The AI backend returned an error.');
                            finishStreaming();
                        }
                    } catch (_) {
                        /* non-JSON lines — ignore */
                    }
                }

                return read();
            };

            return read();
        })
        .catch((err) => {
            if (err.name === 'AbortError') {
                if (typingEl.parentNode) typingEl.parentNode.removeChild(typingEl);
                finishStreaming();
                return;
            }
            console.error('CraftPilot:', err);
            if (typingEl.parentNode) typingEl.parentNode.removeChild(typingEl);
            showError('Erreur: ' + err.name + ' — ' + err.message);
            finishStreaming();
        });
};

const finishStreaming = () => {
    state.streaming      = false;
    state.streamAbort    = null;
    dom.input.disabled   = false;
    dom.sendBtn.disabled = false;
    dom.input?.focus();
};

/* ============================================================
   SAVE MESSAGE  (fire-and-forget)
   ============================================================ */
const saveMessage = (convId, role, content, metadata = '') => {
    Ajax.call([{
        methodname: 'local_craftpilot_manage_messages',
        args: {
            action:          'save',
            conversation_id: convId,
            message_type:    role === 'ai' ? 'ai' : 'user',
            content,
            metadata,
        },
    }])[0].catch(err => console.error('CraftPilot save error:', err));
};

/* ============================================================
   UI HELPERS
   ============================================================ */
const updatePanelTitle = (title) => {
    if (dom.convTitle) dom.convTitle.textContent = title || 'CraftPilot';
};

const scrollBottom = () => {
    const el = dom.messages;
    if (!el) return;
    // Only auto-scroll if the user is within 120px of the bottom.
    // This lets the user scroll up to read earlier content without being
    // dragged back down on every token.
    if (el.scrollHeight - el.scrollTop - el.clientHeight < 120) {
        el.scrollTop = el.scrollHeight;
    }
};

const autoResize = () => {
    const el = dom.input;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
};

const generateUUID = () => {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
        const r = Math.random() * 16 | 0;
        return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
};

const escapeHtml = (s) => {
    if (typeof s !== 'string') return '';
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
};

const stripThinkTags = (s) => {
    if (typeof s !== 'string') return s;
    return s.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
};

const renderMarkdown = (text) => {
    const markedFn = (MarkedLib && MarkedLib.marked) ? MarkedLib.marked : window.marked;
    if (!markedFn) return escapeHtml(text);
    if (!window._cpMarkedConfigured) {
        markedFn.setOptions({ gfm: true, breaks: true });
        window._cpMarkedConfigured = true;
    }
    const raw = markedFn.parse(text);
    return DOMPurify ? DOMPurify.sanitize(raw) : raw;
};

const extractFollowUpQuestions = (text) => {
    state.followUpQuestions = { A: null, B: null, C: null };
    const pattern = /\*\*([ABC])\.\*\*\s+(.+)/g;
    let m;
    while ((m = pattern.exec(text)) !== null) {
        state.followUpQuestions[m[1]] = m[2].trim();
    }
};

const formatTime = (secs) => {
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    const s = Math.floor(secs % 60);
    return [h, m, s].map(n => String(n).padStart(2, '0')).join(':');
};

const svgClose = () =>
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">' +
        '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>' +
    '</svg>';

const svgPlay = () =>
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';

const svgPause = () =>
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">' +
        '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>' +
    '</svg>';
