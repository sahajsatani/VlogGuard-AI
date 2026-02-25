from concurrent.futures import ThreadPoolExecutor
import threading
import queue
from collections import deque
from uniface import RetinaFace
from uniface import ArcFace
from uniface import BlurFace
from uniface.model_store import set_cache_dir, get_cache_dir
import numpy as np
import cv2
import logging as logger
from moviepy import VideoFileClip
import time
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
set_cache_dir(cur_dir + "/cache/models")
logger.info(f"Cache directory set to: {get_cache_dir()}")
detector = RetinaFace()
recognizer = ArcFace()
blurrer = BlurFace(method="pixelate")


class FaceDatabase:
    def __init__(self):
        self.faces_embeddings = []
        self._embeddings_matrix = None  # Precomputed matrix for fast dot product

    def add_face(self, img_path):
        face_img = cv2.imread(img_path)
        faces = detector.detect(face_img)
        logger.info(f"Found {len(faces)} faces in {img_path}")
        if not faces:
            return False
        for face in faces:
            embedding = recognizer.get_normalized_embedding(face_img, face.landmarks)
            self.faces_embeddings.append(embedding.flatten())
        self._rebuild_matrix()
        return True

    def _rebuild_matrix(self):
        """Stack all embeddings into a single matrix for vectorized comparison."""
        if self.faces_embeddings:
            self._embeddings_matrix = np.stack(self.faces_embeddings, axis=0)  # (N, D)

    def best_scores_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Vectorized similarity: compute all face-vs-DB scores in one matmul.
        embeddings: (F, D)  — F faces detected in this frame
        returns:    (F,)    — best similarity score per face
        """
        if self._embeddings_matrix is None or len(embeddings) == 0:
            return np.zeros(len(embeddings))
        # (F, D) @ (D, N) → (F, N), then max across DB axis
        scores = embeddings @ self._embeddings_matrix.T          # (F, N)
        return scores.max(axis=1)                                 # (F,)

    def blur_faces(self, frame, threshold=0.6):
        faces = detector.detect(frame)
        if not faces:
            return frame

        # Batch-extract all embeddings at once
        embeddings = np.stack([
            recognizer.get_normalized_embedding(frame, f.landmarks).flatten()
            for f in faces
        ], axis=0)  # (F, D)

        best_scores = self.best_scores_batch(embeddings)          # (F,) — single matmul

        face_to_blur = [f for f, score in zip(faces, best_scores) if score < threshold]
        blurrer.anonymize(frame, face_to_blur, inplace=True)
        return frame


# ── Tracking helpers ────────────────────────────────────────────────────────

def iou(boxA, boxB):
    """Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (aA + aB - inter)


class FrameTracker:
    """
    Light-weight tracker: re-uses face identities from the previous keyframe
    as long as boxes are stable (high IoU). Runs detection+recognition only
    on keyframes; intermediate frames just re-apply the last blur decision.
    """
    def __init__(self, keyframe_interval: int = 5, iou_threshold: float = 0.45):
        self.interval      = keyframe_interval   # run full detection every N frames
        self.iou_threshold = iou_threshold
        self._last_bboxes: list       = []        # bboxes from last keyframe
        self._last_blur_mask: list[bool] = []     # True = should blur
        self._frame_counter            = 0

    def should_run_detection(self) -> bool:
        return self._frame_counter % self.interval == 0

    def update_keyframe(self, faces, blur_mask: list[bool]):
        self._last_bboxes    = [f.bbox for f in faces]
        self._last_blur_mask = blur_mask
        self._frame_counter += 1

    def get_cached_blur_bboxes(self, faces) -> list | None:
        """
        For non-keyframes: match current detections to cached identities via IoU.
        Returns list of bboxes to blur, or None if tracking fails (force keyframe).
        """
        self._frame_counter += 1
        if not self._last_bboxes or not faces:
            return [f.bbox for f in faces] if faces else []

        current_bboxes = [f.bbox for f in faces]

        # If face count changed significantly → force re-detection
        if abs(len(current_bboxes) - len(self._last_bboxes)) > 1:
            return None

        blur_bboxes = []
        for curr_box in current_bboxes:
            # Find best matching cached box
            best_iou, best_idx = 0.0, -1
            for idx, prev_box in enumerate(self._last_bboxes):
                score = iou(curr_box, prev_box)
                if score > best_iou:
                    best_iou, best_idx = score, idx

            if best_iou >= self.iou_threshold and best_idx != -1:
                if self._last_blur_mask[best_idx]:   # cached decision: blur
                    blur_bboxes.append(curr_box)
            else:
                return None  # Can't match → force keyframe

        return blur_bboxes

def _delete_temp_files(file_paths: list[str]):
    time.sleep(60*10)
    for file in file_paths:
        try:
            if os.path.exists(file):
                os.remove(file)
        except OSError as e:
            e.with_traceback()
            pass
    
def process_raw_video(
    video_path: str,
    face_paths: list[str],
    output_path: str,
    temp_path: str,
    keyframe_interval: int = 5,      # ↑ = faster but less responsive to new faces
    num_workers: int = 8,            # parallel recognition threads
    progress_callback: callable = None,
) -> str | None:
    try:
        # ── Build DB ──────────────────────────────────────────────────────────
        db = FaceDatabase()
        for path in face_paths:
            db.add_face(path)
        logger.info(f"Face database: {len(db.faces_embeddings)} embeddings")

        # ── Open video ───────────────────────────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        fps    = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            logger.error("Could not open VideoWriter")
            return None

        logger.info(f"Total frames: {total}")

        tracker = FrameTracker(keyframe_interval=keyframe_interval)
        frame_idx = 0

        # ── Read & decode frames in a background thread ───────────────────────
        read_queue: queue.Queue = queue.Queue(maxsize=16)

        def reader_thread():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    read_queue.put(None)
                    break
                read_queue.put(frame)

        threading.Thread(target=reader_thread, daemon=True).start()

        # ── Main processing loop ──────────────────────────────────────────────
        while True:
            frame = read_queue.get()
            if frame is None:
                break

            if tracker.should_run_detection():
                # ── Keyframe: full detection + recognition ────────────────
                faces = detector.detect(frame)

                if faces: 
                    embeddings = np.stack([
                        recognizer.get_normalized_embedding(frame, f.landmarks).flatten()
                        for f in faces
                    ], axis=0)
                    best_scores = db.best_scores_batch(embeddings)
                    blur_mask   = (best_scores < 0.6).tolist()
                    tracker.update_keyframe(faces, blur_mask)

                    face_to_blur = [f for f, do_blur in zip(faces, blur_mask) if do_blur]
                    blurrer.anonymize(frame, face_to_blur, inplace=True)
                else:
                    tracker.update_keyframe([], [])

            else:
                # ── Intermediate frame: try cheap tracking ─────────────────
                faces = detector.detect(frame)          # detection is still needed
                blur_bboxes = tracker.get_cached_blur_bboxes(faces)

                if blur_bboxes is None:
                    # Tracking failed → fall back to full recognition
                    if faces:
                        embeddings = np.stack([
                            recognizer.get_normalized_embedding(frame, f.landmarks).flatten()
                            for f in faces
                        ], axis=0)
                        best_scores = db.best_scores_batch(embeddings)
                        blur_mask   = (best_scores < 0.6).tolist()
                        tracker.update_keyframe(faces, blur_mask)
                        face_to_blur = [f for f, do_blur in zip(faces, blur_mask) if do_blur]
                        blurrer.anonymize(frame, face_to_blur, inplace=True)
                else:
                    # Apply blur directly using cached bbox decisions
                    for bbox in blur_bboxes:
                        x1, y1, x2, y2 = map(int, bbox)
                        roi = frame[y1:y2, x1:x2]
                        if roi.size > 0:
                            # Pixelate: downsample → upsample
                            small = cv2.resize(roi, (roi.shape[1]//10 or 1, roi.shape[0]//10 or 1))
                            frame[y1:y2, x1:x2] = cv2.resize(small, (roi.shape[1], roi.shape[0]),
                                                               interpolation=cv2.INTER_NEAREST)

            out.write(frame)
            frame_idx += 1
            if frame_idx % 10 == 0:
                pct = int((frame_idx / total) * 100) if total > 0 else 0
                msg = f"Processing frame {frame_idx}/{total}..."
                if progress_callback:
                    progress_callback(pct, msg)
                if frame_idx % 100 == 0:
                    logger.info(msg)

        cap.release()
        out.release()
        logger.info(f"Done → {output_path}")

        # original_clip = VideoFileClip(video_path)
        # processed_clip = VideoFileClip(output_path)

        # final_clip = processed_clip.with_audio(original_clip.audio)
        # final_clip.write_videofile(temp_path, codec='libx264', audio_codec='aac')

        # original_clip.close()
        # processed_clip.close()
        # final_clip.close()

        print(f"Deleting temp files — {temp_path}, {output_path}")
        thread_delete = threading.Thread(
            target=_delete_temp_files,
            args=([temp_path, output_path],),
            daemon=True
        )
        thread_delete.start()

        # return temp_path
        return output_path

    except Exception as e:
        logger.exception(f"Error processing video: {e}")
        return None
