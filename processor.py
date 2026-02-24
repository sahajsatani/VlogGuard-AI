"""
processor.py – VlogGuard AI video processor
  • Detects all faces per frame using face_recognition
  • Blurs every face whose encoding does NOT match ANY of the vlogger's reference photos
  • Accepts multiple reference face images for higher precision matching
  • Detects number plates using OpenCV's Haar cascade and blurs them
  • Reconstructs the video at original FPS & resolution
"""
import cv2
from uniface.detection import RetinaFace
from uniface.recognition import ArcFace
from uniface.privacy import BlurFace
import torch
from huggingface_hub import hf_hub_download
from moviepy import VideoFileClip
from ultralytics import YOLO
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
detector = RetinaFace()
recognizer = ArcFace()
blurrer = BlurFace(method="pixelate")

class FaceDatabase:
    def __init__(self):
        self.faces_embeddings = []
    
    def add_face(self, img_path):
        face_img = cv2.imread(img_path)
        faces = detector.detect(face_img)
        logger.info(f"Found {len(faces)} faces in {img_path}")
        if not faces:
            return False
        for face in faces:
            embedding = recognizer.get_normalized_embedding(face_img, face.landmarks)
            self.faces_embeddings.append(embedding)
        return True
    
    def blur_faces(self, frame, thresold=0.6):
        faces = detector.detect(frame)

        result = []
        face_to_blur=[]
        for i,face in enumerate(faces):
            embedding = recognizer.get_normalized_embedding(frame, face.landmarks)
            
            best_score = 0
            for ref_emb in self.faces_embeddings:
                similarity = np.dot(embedding.flatten(), ref_emb.flatten()) #/ (np.linalg.norm(embedding) * np.linalg.norm(ref_emb))
                best_score = max(best_score, similarity)
            
            if best_score >= thresold:
                result.append(face.bbox)
            else:
                face_to_blur.append(face)
        
        # blur faces
        blurrer.anonymize(frame, face_to_blur,inplace=True)

        return frame

class FaceDatabaseYolo:
    def __init__(self):
        self.faces_embeddings = []
        self.last_face_to_blur = []
        self.frame_count = 0
    
    def add_face(self, img_path):
        face_img = cv2.imread(img_path)
        faces = detector.detect(face_img)
        logger.info(f"Found {len(faces)} faces in {img_path}")
        if not faces:
            return False
        for face in faces:
            embedding = recognizer.get_normalized_embedding(face_img, face.landmarks)
            self.faces_embeddings.append(embedding)
        return True

    def blur_faces(self, frame, thresold=0.6, skip_frames=5):
        self.frame_count += 1
        
        # Only run AI every 'skip_frames'
        if self.frame_count % skip_frames != 0:
            if self.last_face_to_blur:
                blurrer.anonymize(frame, self.last_face_to_blur, inplace=True)
            return frame

        faces = detector.detect(frame)
        if not faces:
            self.last_face_to_blur = []
            return frame

        # Convert list of reference embeddings to a single NumPy Matrix (do this once in add_face)
        ref_matrix = np.array(self.faces_embeddings) 

        face_to_blur = []
        for face in faces:
            embedding = recognizer.get_normalized_embedding(frame, face.landmarks).flatten()
            
            # FAST MATRIX SEARCH: Compare 1 face against ALL database faces at once
            similarities = np.dot(ref_matrix, embedding)
            best_score = np.max(similarities)
            
            if best_score < thresold:
                face_to_blur.append(face)
        
        self.last_face_to_blur = face_to_blur
        blurrer.anonymize(frame, face_to_blur, inplace=True)
        return frame

def process_video(
    video_path: str,
    face_paths: list[str],
    output_path: str,
) -> None:
    try:
        db = FaceDatabase()
        for path in face_paths:
            db.add_face(path)
        logger.info(f"Face database created with {len(db.faces_embeddings)} embeddings")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 3. Use H.264 Codec (Best for quality/web compatibility)
        # 'avc1' is the FourCC for H.264
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            logger.error("Error: Could not open video writer.")
            return None 
        logger.info(f"Total frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
        while cap.isOpened():
            logger.info(f"Processing frame {cap.get(cv2.CAP_PROP_POS_FRAMES)} / {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = db.blur_faces(frame)

            out.write(frame)
        
        cap.release()
        out.release()
        logger.info(f"Video processed successfully: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error processing video: {e}")
        e.with_traceback()
        return None

class VlogFaceBlurrer:
    def __init__(self, target_face_paths=None, threshold=0.5):
        # 1. AUTO-DOWNLOAD the specialized face model weights
        print("Checking for face detection weights...")
        model_path = hf_hub_download(
            repo_id="arnabdhar/YOLOv8-Face-Detection", 
            filename="model.pt"
        )
        self.detector = YOLO(model_path)
        
        # 2. Recognition Setup
        self.recognizer = ArcFace()
        self.vlogger_embeddings = [] # List to store multiple vlogger embeddings
        self.threshold = threshold
        
        # 3. Tracking cache (Real-time speed boost)
        self.is_vlogger_cache = {} # {track_id: bool}
        
        if target_face_paths:
            self.add_vloggers(target_face_paths)

    def add_vloggers(self, img_paths):
        """Register multiple Vlogger faces so they DON'T get blurred."""
        if isinstance(img_paths, str):
            img_paths = [img_paths]
            
        for path in img_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Could not read image at {path}")
                continue
                
            results = self.detector(img, verbose=False)
            if results and len(results[0].boxes) > 0:
                # Get embedding for the reference face
                emb = self.recognizer.get_normalized_embedding(img, None).flatten()
                self.vlogger_embeddings.append(emb)
                print(f"Vlogger face from {path} registered successfully.")
            else:
                print(f"Warning: No face detected in {path}!")

    def process_frame(self, frame):
        # Run tracking with face-specific model
        results = self.detector.track(frame, persist=True, verbose=False, conf=0.4)
        
        if not results or not results[0].boxes:
            return frame

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else [None] * len(boxes)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            
            # Check cache first (Crucial for 30fps performance)
            if track_id is not None and track_id in self.is_vlogger_cache:
                is_vlogger = self.is_vlogger_cache[track_id]
            else:
                # New face detected: check against Vlogger embeddings
                is_vlogger = self._is_this_a_vlogger(frame, box)
                if track_id is not None:
                    self.is_vlogger_cache[track_id] = is_vlogger

            # Only blur if it's NOT one of the vloggers
            if not is_vlogger:
                self._apply_face_blur(frame, box)

        return frame

    def _is_this_a_vlogger(self, frame, box):
        if not self.vlogger_embeddings: 
            return False
        try:
            # Extract embedding for current detected face
            curr_emb = self.recognizer.get_normalized_embedding(frame, None).flatten()
            
            # Check against all registered vlogger embeddings
            for v_emb in self.vlogger_embeddings:
                similarity = np.dot(v_emb, curr_emb)
                if similarity > self.threshold:
                    return True
            return False
        except:
            return False

    def _apply_face_blur(self, frame, box):
        x1, y1, x2, y2 = box
        # Padding coordinates to ensure full face coverage
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1-10), max(0, y1-10)
        x2, y2 = min(w, x2+10), min(h, y2+10)
        
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size > 0:
            # Strong Gaussian Blur
            blurred = cv2.GaussianBlur(face_roi, (99, 99), 30)
            frame[y1:y2, x1:x2] = blurred

class VlogFaceBlurrerAudio:
    def __init__(self, target_face_paths=None, threshold=0.5):
        # 1. AUTO-DOWNLOAD the specialized face model weights
        print("Checking for face detection weights...")
        model_path = hf_hub_download(
            repo_id="arnabdhar/YOLOv8-Face-Detection", 
            filename="model.pt"
        )
        if os.path.exists(model_path):
            print(f"Model weights found at {model_path}")
        else:
            print("Error: Model weights not found after download!")
            return
        self.detector = YOLO(model_path)
        
        # 2. Recognition Setup
        self.recognizer = ArcFace()
        self.vlogger_embeddings = [] 
        self.threshold = threshold
        
        # 3. Tracking cache (Real-time speed boost)
        self.is_vlogger_cache = {} 
        
        if target_face_paths:
            self.add_vloggers(target_face_paths)

    def add_vloggers(self, img_paths):
        """Register multiple Vlogger faces so they DON'T get blurred."""
        if isinstance(img_paths, str):
            img_paths = [img_paths]
            
        for path in img_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Could not read image at {path}")
                continue
                
            results = self.detector(img, verbose=False)
            if results and len(results[0].boxes) > 0:
                emb = self.recognizer.get_normalized_embedding(img, None).flatten()
                self.vlogger_embeddings.append(emb)
                print(f"Vlogger face from {path} registered successfully.")
            else:
                print(f"Warning: No face detected in {path}!")

    def process_frame(self, frame):
        # Run tracking with face-specific model
        results = self.detector.track(frame, persist=True, verbose=False, conf=0.4)
        
        if not results or not results[0].boxes:
            return frame

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else [None] * len(boxes)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            
            if track_id is not None and track_id in self.is_vlogger_cache:
                is_vlogger = self.is_vlogger_cache[track_id]
            else:
                is_vlogger = self._is_this_a_vlogger(frame, box)
                if track_id is not None:
                    self.is_vlogger_cache[track_id] = is_vlogger

            if not is_vlogger:
                self._apply_face_blur(frame, box)

        return frame

    def _is_this_a_vlogger(self, frame, box):
        if not self.vlogger_embeddings: 
            return False
        try:
            curr_emb = self.recognizer.get_normalized_embedding(frame, None).flatten()
            for v_emb in self.vlogger_embeddings:
                similarity = np.dot(v_emb, curr_emb)
                if similarity > self.threshold:
                    return True
            return False
        except:
            return False

    def _apply_face_blur(self, frame, box):
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1-10), max(0, y1-10)
        x2, y2 = min(w, x2+10), min(h, y2+10)
        
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size > 0:
            blurred = cv2.GaussianBlur(face_roi, (99, 99), 30)
            frame[y1:y2, x1:x2] = blurred

def process_video_with_audio(video_path, face_paths, output_path):
    # 1. Initialize Processor
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    app = VlogFaceBlurrerAudio()
    for img in face_paths:
        app.add_vloggers(img)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 2. Setup Temporary Video Writer (Visuals only)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Starting processing: {total_frames} frames...")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = app.process_frame(frame)
        out.write(processed_frame)
        
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    out.release()

    # temp_output = output_path.replace(".mp4", "_temp.mp4")
    # 3. Re-attach Audio using MoviePy
    print("Re-attaching audio stream...")
    try:
        original_clip = VideoFileClip(video_path)
        processed_clip = VideoFileClip(output_path)
        
        # Combine blurred video with original audio
        final_clip = processed_clip.set_audio(original_clip.audio)
        
        # Write final file with high-quality H.264 codec
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        # Cleanup temp file
        processed_clip.close()
        original_clip.close()
        # if os.path.exists(output_path):
        #     os.remove(output_path)
            
        print(f"Successfully saved blurred video with audio to: {output_path}")
        
    except Exception as e:
        print(f"Error during audio attachment: {e}")
        print(f"The silent blurred video is still available at: {output_path}")
        e.with_traceback()

    return output_path



# from __future__ import annotations
# import os
# import cv2
# import numpy as np

# try:
#     import face_recognition
#     FACE_RECOGNITION_AVAILABLE = True
# except ImportError:
#     FACE_RECOGNITION_AVAILABLE = False
#     print("[processor] face_recognition not installed – falling back to OpenCV face detector")

# # Path to OpenCV's built-in plate cascade
# _CASCADE_PATHS = [
#     cv2.data.haarcascades + "haarcascade_russian_plate_number.xml",
#     cv2.data.haarcascades + "haarcascade_licence_plate_rus_16stages.xml",
# ]
# _PLATE_CASCADE: cv2.CascadeClassifier | None = None
# for _p in _CASCADE_PATHS:
#     if os.path.exists(_p):
#         _PLATE_CASCADE = cv2.CascadeClassifier(_p)
#         break

# # OpenCV face cascade fallback
# _FACE_CASCADE = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )


# # ─────────────────────────────────────────────────────────────────────────────
# def _blur_region(frame: np.ndarray, x: int, y: int, w: int, h: int,
#                  strength: int = 51) -> np.ndarray:
#     """Apply a strong Gaussian blur to the rectangle (x,y,w,h)."""
#     roi = frame[y : y + h, x : x + w]
#     if roi.size == 0:
#         return frame
#     # Make kernel odd and large
#     k = max(strength | 1, 51)
#     blurred = cv2.GaussianBlur(roi, (k, k), 30)
#     frame[y : y + h, x : x + w] = blurred
#     return frame


# # ─────────────────────────────────────────────────────────────────────────────
# def _encode_references(face_paths: list[str]) -> list:
#     """
#     Return a combined list of face encodings from ALL provided reference images.
#     More images → more encodings → higher matching recall.
#     """
#     if not FACE_RECOGNITION_AVAILABLE:
#         return []
#     all_encodings: list = []
#     for path in face_paths:
#         try:
#             img = face_recognition.load_image_file(path)
#             encodings = face_recognition.face_encodings(img)
#             all_encodings.extend(encodings)
#             if not encodings:
#                 print(f"[processor] No face found in reference image: {path}")
#         except Exception as e:
#             print(f"[processor] Could not encode {path}: {e}")
#     return all_encodings


# # ─────────────────────────────────────────────────────────────────────────────
# def _process_frame_face_recognition(
#     frame: np.ndarray,
#     ref_encodings: list,
#     tolerance: float = 0.55,
# ) -> np.ndarray:
#     """
#     Detect faces; blur those that DON'T match any ref encoding.
#     """
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     locations = face_recognition.face_locations(rgb, model="hog")
#     if not locations:
#         return frame

#     encodings = face_recognition.face_encodings(rgb, locations)

#     for (top, right, bottom, left), enc in zip(locations, encodings):
#         if ref_encodings:
#             matches = face_recognition.compare_faces(ref_encodings, enc, tolerance=tolerance)
#             if any(matches):
#                 continue   # vlogger – leave unblurred
#         x, y, w, h = left, top, right - left, bottom - top
#         frame = _blur_region(frame, x, y, w, h)

#     return frame


# def _process_frame_opencv_faces(frame: np.ndarray) -> np.ndarray:
#     """
#     Fallback: blur ALL detected faces (no identity exclusion).
#     """
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
#     for (x, y, w, h) in faces:
#         frame = _blur_region(frame, x, y, w, h)
#     return frame


# def _process_frame_plates(frame: np.ndarray) -> np.ndarray:
#     """Detect number plates with Haar cascade and blur them."""
#     if _PLATE_CASCADE is None:
#         return frame
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     plates = _PLATE_CASCADE.detectMultiScale(
#         gray, scaleFactor=1.05, minNeighbors=3, minSize=(60, 20)
#     )
#     for (x, y, w, h) in plates:
#         # Expand region slightly for cleaner blur
#         pad = 4
#         x1 = max(0, x - pad)
#         y1 = max(0, y - pad)
#         x2 = min(frame.shape[1], x + w + pad)
#         y2 = min(frame.shape[0], y + h + pad)
#         frame = _blur_region(frame, x1, y1, x2 - x1, y2 - y1, strength=61)
#     return frame


# # ─────────────────────────────────────────────────────────────────────────────
# def process_video(
#     video_path: str,
#     face_paths: list[str],
#     output_path: str,
# ) -> None:

#     """
#     Main entry point called by app.py.

#     Args:
#         video_path:  Path to uploaded input video.
#         face_paths:  List of paths to vlogger reference face images (1 or more).
#         output_path: Where to write the processed video (.mp4).
#     """

    # cap = cv2.VideoCapture(video_path)
    # if not cap.isOpened():
    #     raise RuntimeError(f"Cannot open video: {video_path}")

    # fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # # Build combined reference encodings from ALL uploaded face images
    # ref_encodings = _encode_references(face_paths) if FACE_RECOGNITION_AVAILABLE else []
    # print(f"[processor] Loaded {len(ref_encodings)} reference encoding(s) from {len(face_paths)} image(s)")

    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     # ── Face blur ─────────────────────────────────────────────────────
    #     if FACE_RECOGNITION_AVAILABLE:
    #         frame = _process_frame_face_recognition(frame, ref_encodings)
    #     else:
    #         # Fallback: OpenCV cascade, no identity exclusion
    #         frame = _process_frame_opencv_faces(frame)

    #     # ── Plate blur ────────────────────────────────────────────────────
    #     frame = _process_frame_plates(frame)

    #     writer.write(frame)

    # cap.release()
    # writer.release()

    # if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
    #     raise RuntimeError("Output video was not written correctly.")
