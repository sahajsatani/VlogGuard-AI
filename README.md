# 🛡️ VlogGuard AI
**Privacy-First Video Anonymization for Creators.**

VlogGuard AI automatically detects and blurs bystanders in your videos while keeping you crystal clear. Using state-of-the-art face recognition, it identifies you based on reference photos and blurs everyone else, ensuring privacy without sacrificing the quality of your content.

---

## ✨ Key Features

- **🎯 Smart Selective Blurring**: Blurs all faces *except* the creator's.
- **📸 Multi-Photo Reference**: Upload multiple angles of yourself for high-precision identity matching.
- **⚡ Async Processing**: Handles long videos in the background with real-time progress tracking.
- **🔒 Privacy-Focused**: Processed videos are stored temporarily and automatically deleted after 10 minutes.
- **🌐 Web Interface**: Clean, modern, and responsive 4-step upload flow.
- **🐢 Free Tier Optimized**: Lightweight processing tailored for creator-first accessibility.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **AI/ML**: [Uniface](https://github.com/sahajsatani/uniface) (RetinaFace for detection, ArcFace for recognition)
- **Computer Vision**: OpenCV
- **Frontend**: Vanilla JS, Modern CSS (Glassmorphism & Vibrant UI)

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Virtual Environment (recommended)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/VlogGuard-AI.git
cd VlogGuard-AI

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the App
```bash
# Set your preferred video size limit (in MB)
export VIDEO_SIZE=50

# Start the Flask server
python3 app.py
```
Open `http://localhost:8000` in your browser.

---

## 📝 Usage Flow

1. **Step 1: Upload Video** – Select your vlog (MP4, MOV, AVI).
2. **Step 2: Provide Reference** – Upload 1-3 clear photos of your face.
3. **Step 3: AI Magic** – Watch the real-time progress bar as the AI blurs bystanders.
4. **Step 4: Download** – Grab your protected video!

---

## ⚠️ Current Limitations
- **Audio Support**: The current free-tier processing is "Slow & Silent" (video only). Full audio preservation is in the roadmap!
- **Model Download**: On the first run, the AI models (~200MB) will be downloaded to the `/cache` directory.

---

## 🤝 Contributing
Contributions are welcome! Please feel free to subimt a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.