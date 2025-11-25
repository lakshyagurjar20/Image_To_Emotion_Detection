# Emotion Detection Web Application

A modern Flask-based web application for detecting emotions from images and videos using pose analysis with **MediaPipe** and **Machine Learning**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.8-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-red)

---

## Features

- **Image Analysis** - Upload images to detect emotions based on body pose
- **Video Processing** - Frame-by-frame emotion detection with real-time analysis
- **Emotion Distribution** - Detailed statistics and visualization for video analysis
- **Modern UI** - Beautiful, responsive web interface with drag-and-drop support
- **Real-time Processing** - Fast emotion detection with pose landmarks visualization
- **Confidence Scores** - Get prediction confidence for each detection

---

## Supported Emotions

The system can detect four primary emotions:

| Emotion     | Description                        |
| ----------- | ---------------------------------- |
| **Angry**   | Aggressive or tense body posture   |
| **Happy**   | Open, relaxed, and positive stance |
| **Neutral** | Balanced and calm posture          |
| **Sad**     | Slumped or withdrawn body language |

---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**

```bash
git clone <your-repo-url>
cd Shiv_project
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Generate model files (First time only):**

```bash
python create_placeholder_models.py
```

4. **Run the application:**

```bash
python app.py
```

5. **Open your browser:**

```
http://localhost:5000
```

---

## Usage Guide

### Upload an Image

1. Click **"Browse Files"** or drag and drop an image
2. Supported formats: **JPG, JPEG, PNG**
3. Click **"Analyze Emotion"**
4. View results with:
   - Detected emotion
   - Confidence percentage
   - Original vs. Processed comparison
   - Pose landmarks visualization

### Upload a Video

1. Click **"Browse Files"** or drag and drop a video
2. Supported formats: **MP4, AVI, MOV, MKV**
3. Click **"Analyze Emotion"**
4. View results with:
   - Dominant emotion across all frames
   - Emotion distribution chart
   - Total frames processed
   - Processed video with overlays

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

---

## Project Structure

```
Shiv_project/
├── app.py                          # Flask web application
├── image_to_mood_dec.ipynb         # Training notebook (use in Google Colab)
├── create_placeholder_models.py    # Script to create dummy models
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── templates/
│   └── index.html                  # Web interface
├── static/
│   └── style.css                   # Styling
├── uploads/                        # User uploaded files (auto-created)
├── emotion_classifier_rf.pkl       # Trained model
└── scaler.pkl                      # Trained scaler
```

---

## Training Data Requirements

Your H5 files must contain:

- **48 features per sample**: MediaPipe pose landmarks (16 keypoints × 3 coordinates)
- **Keypoints used**: Head, shoulders, elbows, wrists, hips, knees, ankles, feet
- **Labels**: Inferred from filename (e.g., `person_Angry_001.h5`)

---

## Troubleshooting

### Error: "Model not loaded"

**Cause:** Models are missing or not properly generated.

**Solution:** Run `python create_placeholder_models.py` to generate the required model files.

### Port Already in Use

```bash
# Change port in app.py
app.run(port=5001)
```

### Video Processing Issues

- Ensure **ffmpeg** is installed on your system
- Check video codec compatibility
- Try converting video to MP4 format

### Low Accuracy

**Solution:**

- Ensure you have enough training samples (recommended: 1000+ per emotion)
- Use diverse pose data
- Train longer or adjust hyperparameters in the notebook

---

## Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is created for educational purposes.

---

## Author

Created using **MediaPipe**, **scikit-learn**, and **Flask**

---

## Acknowledgments

- **MediaPipe** by Google for pose detection
- **Flask** community for excellent documentation
- **scikit-learn** for machine learning tools

---

## Support

For questions or issues, please open an issue in the GitHub repository.
