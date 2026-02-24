/* ══════════════════════════════════════════════════════
   VlogGuard AI – main.js
   Handles: drag-and-drop, 100 MB validation, stepper,
            fetch to /process, progress bar, result display
   ══════════════════════════════════════════════════════ */

// ── Navbar scroll effect ────────────────────────────────
const navbar = document.querySelector(".navbar");
window.addEventListener("scroll", () => {
    navbar.style.background =
        window.scrollY > 60 ? "rgba(5,8,22,.95)" : "rgba(5,8,22,.7)";
});

// ── Smooth anchor scroll ────────────────────────────────
document.querySelectorAll('a[href^="#"]').forEach((a) => {
    a.addEventListener("click", (e) => {
        const target = document.querySelector(a.getAttribute("href"));
        if (target) { e.preventDefault(); target.scrollIntoView({ behavior: "smooth" }); }
    });
});


// ══════════════════════════════════════════════════════
// State
// ══════════════════════════════════════════════════════
let videoFile = null;
let faceFile = null;


// ══════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════
const MAX_VIDEO_BYTES = 100 * 1024 * 1024; // 100 MB

function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function showError(el, msg) {
    el.textContent = msg;
    el.classList.remove("hidden");
}
function hideError(el) {
    el.textContent = "";
    el.classList.add("hidden");
}

// ── Stepper ────────────────────────────────────────────
function setStep(n) {
    // Steps: 1=video, 2=face, 3=processing, 4=result
    const ids = ["step-video", "step-face", "step-processing", "step-result"];
    ids.forEach((id, i) => {
        const el = document.getElementById(id);
        if (i + 1 === n) el.classList.remove("hidden");
        else el.classList.add("hidden");
    });

    // Stepper circles
    for (let i = 1; i <= 4; i++) {
        const st = document.getElementById(`st-${i}`);
        st.classList.toggle("active", i === n);
        st.classList.toggle("done", i < n);
    }

    // Step lines
    const lines = document.querySelectorAll(".step-line");
    lines.forEach((l, i) => l.classList.toggle("filled", i < n - 1));

    // Scroll into view
    document.getElementById("upload-section").scrollIntoView({ behavior: "smooth" });
}


// ══════════════════════════════════════════════════════
// Step 1 – Video Upload
// ══════════════════════════════════════════════════════
const videoDrop = document.getElementById("video-drop");
const videoInput = document.getElementById("video-input");
const videoError = document.getElementById("video-error");
const videoNext = document.getElementById("video-next");
const videoPreview = document.getElementById("video-preview");
const videoContent = document.getElementById("video-drop-content");
const videoName = document.getElementById("video-name");
const videoSize = document.getElementById("video-size");
const videoRemove = document.getElementById("video-remove");

function setVideoFile(file) {
    hideError(videoError);

    if (!file) return;

    // MIME type check
    const allowed = ["video/mp4", "video/quicktime", "video/avi", "video/webm",
        "video/x-matroska", "video/x-msvideo"];
    const ext = file.name.split(".").pop().toLowerCase();
    const allowedExt = ["mp4", "mov", "avi", "webm", "mkv"];

    if (!allowed.includes(file.type) && !allowedExt.includes(ext)) {
        showError(videoError, "❌ Please upload a valid video file (MP4, MOV, AVI, WebM).");
        videoDrop.classList.add("error");
        return;
    }

    // Size check
    if (file.size > MAX_VIDEO_BYTES) {
        showError(
            videoError,
            `❌ File too large: ${formatSize(file.size)}. Maximum allowed is 100 MB.`
        );
        videoDrop.classList.add("error");
        videoDrop.classList.remove("success");
        videoFile = null;
        videoNext.disabled = true;
        return;
    }

    videoFile = file;
    videoName.textContent = file.name;
    videoSize.textContent = formatSize(file.size);

    videoContent.classList.add("hidden");
    videoPreview.classList.remove("hidden");
    videoDrop.classList.remove("error", "drag-over");
    videoDrop.classList.add("success");
    videoNext.disabled = false;
}

videoInput.addEventListener("change", () => setVideoFile(videoInput.files[0]));

videoDrop.addEventListener("dragover", (e) => {
    e.preventDefault();
    videoDrop.classList.add("drag-over");
});
videoDrop.addEventListener("dragleave", () => videoDrop.classList.remove("drag-over"));
videoDrop.addEventListener("drop", (e) => {
    e.preventDefault();
    videoDrop.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) setVideoFile(file);
});
videoDrop.addEventListener("click", (e) => {
    if (e.target !== videoRemove && !videoRemove.contains(e.target)) {
        if (!videoFile) videoInput.click();
    }
});

videoRemove.addEventListener("click", (e) => {
    e.stopPropagation();
    videoFile = null;
    videoInput.value = "";
    videoContent.classList.remove("hidden");
    videoPreview.classList.add("hidden");
    videoDrop.classList.remove("success", "error");
    videoNext.disabled = true;
    hideError(videoError);
});

videoNext.addEventListener("click", () => {
    if (!videoFile) return;
    setStep(2);
});


// ══════════════════════════════════════════════════════
// Step 2 – Multiple Face Photo Upload
// ══════════════════════════════════════════════════════
const faceInput       = document.getElementById("face-input");
const faceError       = document.getElementById("face-error");
const faceNext        = document.getElementById("face-next");
const faceDrop        = document.getElementById("face-drop");
const facesContainer  = document.getElementById("faces-container");
const facesGrid       = document.getElementById("faces-grid");
const facesCountEl    = document.getElementById("faces-count");
const faceAddMore     = document.getElementById("face-add-more");
const faceBack        = document.getElementById("face-back");
const faceBrowseBtn   = document.getElementById("face-browse-btn");

// Array of { file, objectUrl }
let faceFiles = [];

function updateFacesUI() {
    const n = faceFiles.length;
    facesCountEl.textContent = `${n} photo${n !== 1 ? "s" : ""} added`;
    faceNext.disabled = n === 0;
    if (n === 0) {
        faceDrop.classList.remove("hidden");
        facesContainer.classList.add("hidden");
    } else {
        faceDrop.classList.add("hidden");
        facesContainer.classList.remove("hidden");
    }
}

function renderFaceGrid() {
    facesGrid.innerHTML = "";
    faceFiles.forEach(({ file, objectUrl }, idx) => {
        const tile = document.createElement("div");
        tile.className = "face-tile";
        tile.innerHTML = `
            <img src="${objectUrl}" alt="${file.name}" class="face-tile-img" />
            <div class="face-tile-info">
                <span class="face-tile-name">${file.name.length > 18 ? file.name.slice(0,16)+"…" : file.name}</span>
                <span class="face-tile-size">${formatSize(file.size)}</span>
            </div>
            <button class="face-tile-remove" data-idx="${idx}" title="Remove">✕</button>
        `;
        tile.querySelector(".face-tile-remove").addEventListener("click", (e) => {
            const i = parseInt(e.currentTarget.getAttribute("data-idx"), 10);
            URL.revokeObjectURL(faceFiles[i].objectUrl);
            faceFiles.splice(i, 1);
            renderFaceGrid();
            updateFacesUI();
        });
        facesGrid.appendChild(tile);
    });
}

function addFaceImages(fileList) {
    hideError(faceError);
    const allowed = ["image/jpeg", "image/png", "image/webp"];
    let rejected = 0;
    Array.from(fileList).forEach((file) => {
        if (!allowed.includes(file.type)) { rejected++; return; }
        const isDupe = faceFiles.some(f => f.file.name === file.name && f.file.size === file.size);
        if (isDupe) return;
        faceFiles.push({ file, objectUrl: URL.createObjectURL(file) });
    });
    if (rejected > 0) {
        showError(faceError, `❌ ${rejected} file(s) skipped — only JPG, PNG, and WebP are accepted.`);
    }
    renderFaceGrid();
    updateFacesUI();
}

faceBrowseBtn.addEventListener("click", () => faceInput.click());
faceAddMore.addEventListener("click", () => faceInput.click());
faceInput.addEventListener("change", () => {
    if (faceInput.files.length) addFaceImages(faceInput.files);
    faceInput.value = "";
});

// Drag & drop on initial drop zone
faceDrop.addEventListener("dragover", (e) => { e.preventDefault(); faceDrop.classList.add("drag-over"); });
faceDrop.addEventListener("dragleave", () => faceDrop.classList.remove("drag-over"));
faceDrop.addEventListener("drop", (e) => {
    e.preventDefault();
    faceDrop.classList.remove("drag-over");
    if (e.dataTransfer.files.length) addFaceImages(e.dataTransfer.files);
});
faceDrop.addEventListener("click", (e) => {
    if (e.target !== faceBrowseBtn) faceInput.click();
});

// Drag & drop on grid container (add more by dropping)
facesContainer.addEventListener("dragover", (e) => { e.preventDefault(); facesContainer.classList.add("drag-over-grid"); });
facesContainer.addEventListener("dragleave", () => facesContainer.classList.remove("drag-over-grid"));
facesContainer.addEventListener("drop", (e) => {
    e.preventDefault();
    facesContainer.classList.remove("drag-over-grid");
    if (e.dataTransfer.files.length) addFaceImages(e.dataTransfer.files);
});

faceBack.addEventListener("click", () => setStep(1));


// ══════════════════════════════════════════════════════
// Step 3 – Processing (fetch + fake progress)
// ══════════════════════════════════════════════════════
const progressBar = document.getElementById("progress-bar");
const progressLabel = document.getElementById("progress-label");

function animateProgress(targetPct, label, durationMs) {
    progressLabel.textContent = label;
    const start = parseFloat(progressBar.style.width) || 0;
    const diff = targetPct - start;
    const steps = 40;
    const delay = durationMs / steps;
    let step = 0;
    return new Promise((res) => {
        const id = setInterval(() => {
            step++;
            progressBar.style.width = `${start + diff * (step / steps)}%`;
            if (step >= steps) { clearInterval(id); res(); }
        }, delay);
    });
}

// Track the current blob URL so we can revoke it on reset
let currentBlobUrl = null;

faceNext.addEventListener("click", async () => {
    if (!videoFile || faceFiles.length === 0) return;

    setStep(3);
    hideError(faceError);

    // 1. Initial Upload Phase
    await animateProgress(10, "Preparing video for upload...", 500);

    const form = new FormData();
    form.append("video", videoFile);
    faceFiles.forEach(({ file }) => form.append("reference_faces", file));
    const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));

    let jobID = null;
    let serverError = null;

    try {
        progressLabel.textContent = "Uploading assets to server...";
        const resp = await fetch("/process", { method: "POST", body: form });
        const json = await resp.json();

        if (!resp.ok) {
            serverError = json.error || `Upload failed (HTTP ${resp.status})`;
        } else {
            jobID = json.job_id;
        }
    } catch (err) {
        serverError = `Upload network error: ${err.message}`;
    }

    if (serverError) {
        setStep(2);
        showError(faceError, `❌ ${serverError}`);
        return;
    }

    // 2. Polling Phase
    let isDone = false;
    let lastProgress = 10;

    const pollStatus = async () => {
        try {
            const resp = await fetch(`/status/${jobID}`);
            if (!resp.ok) throw new Error(`Status check failed (HTTP ${resp.status})`);
            
            const data = await resp.json();
            
            if (data.status === "error") {
                serverError = data.error || "Server processing failed.";
                isDone = true;
            } else if (data.status === "done") {
                await animateProgress(100, "Finalizing...", 500);
                isDone = true;
            } else {
                // map 0-100 progress to the 10-95% range on our UI
                const uiProgress = 10 + (data.progress * 0.85); 
                if (uiProgress > lastProgress) {
                    await animateProgress(uiProgress, data.message || "Processing...", 400);
                    lastProgress = uiProgress;
                }
            }
        } catch (err) {
            console.error("Polling error:", err);
            // We don't stop immediately on a single network jitter during polling
        }
    };

    // Keep polling every 2 seconds until done or error
    while (!isDone && !serverError) {
        await pollStatus();
        if (!isDone) await new Promise(r => setTimeout(r, 5000));
    }

    if (serverError) {
        setStep(2);
        showError(faceError, `❌ ${serverError}`);
        return;
    }

    // 3. Complete Phase
    const dlBtn = document.getElementById("download-btn");
    dlBtn.href  = `/download/${jobID}`;
    dlBtn.setAttribute("download", "vlogguard_processed.mp4");

    setStep(4);
});


// ══════════════════════════════════════════════════════
// Step 4 – Process Another
// ══════════════════════════════════════════════════════
document.getElementById("process-another").addEventListener("click", () => {
    // Revoke existing blob URL to free browser memory
    if (currentBlobUrl) {
        URL.revokeObjectURL(currentBlobUrl);
        currentBlobUrl = null;
    }

    // Revoke all face preview blob URLs
    faceFiles.forEach(({ objectUrl }) => URL.revokeObjectURL(objectUrl));
    faceFiles = [];

    // Reset state
    videoFile = null;

    videoInput.value = "";
    faceInput.value = "";

    videoContent.classList.remove("hidden");
    videoPreview.classList.add("hidden");
    videoDrop.classList.remove("success", "error");
    videoNext.disabled = true;

    // Reset face step
    faceDrop.classList.remove("hidden");
    facesContainer.classList.add("hidden");
    facesGrid.innerHTML = "";
    faceNext.disabled = true;

    hideError(videoError);
    hideError(faceError);

    progressBar.style.width = "0%";
    progressLabel.textContent = "Uploading…";

    setStep(1);
});
