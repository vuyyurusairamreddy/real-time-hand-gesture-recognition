# Hand Gesture Recognition System

**Developer:** VUYYURU SAIRAMREDDY 

A real-time hand gesture recognition application built with Python, OpenCV, and MediaPipe that can recognize four distinct static hand gestures using a standard webcam.

## 🎯 Supported Gestures

- **Open Palm** - All five fingers extended
- **Fist** - All fingers closed
- **Peace Sign** - Index and middle fingers extended (V-sign)
- **Thumbs Up** - Only thumb extended upward
## Demo Video

[Watch the demo video on Google Drive](https://drive.google.com/file/d/10B0cL4PqUrTp6BUTT7R0nJ9XB1MNV/view?usp=sharing)


## 🔧 Technology Justification

### MediaPipe Hands
I chose **Google's MediaPipe Hands** as the primary hand detection and tracking solution for the following reasons:

1. **Accuracy & Robustness**: MediaPipe provides state-of-the-art hand landmark detection with 21 key points, offering precise finger joint locations essential for gesture classification.

2. **Real-time Performance**: Optimized for real-time applications with efficient inference on CPU, ensuring smooth gesture recognition without significant lag.

3. **Easy Integration**: Well-documented Python API with seamless OpenCV integration, reducing development complexity.

4. **Reliability**: Handles various lighting conditions, hand orientations, and partial occlusions better than traditional computer vision approaches.

5. **No Training Required**: Pre-trained models eliminate the need for custom dataset collection and training, making it ideal for rapid prototyping.

### OpenCV
**OpenCV** was selected for video capture and display because:
- Industry standard for computer vision applications
- Excellent webcam integration with `cv2.VideoCapture`
- Comprehensive drawing utilities for visual feedback
- Cross-platform compatibility

## 🧠 Gesture Logic Explanation

The gesture recognition system uses a **rule-based approach** analyzing the positions of 21 hand landmarks provided by MediaPipe:

### Hand Landmark Analysis
- **Landmark 0**: Wrist (reference point)
- **Landmarks 1-4**: Thumb joints
- **Landmarks 5-8**: Index finger joints
- **Landmarks 9-12**: Middle finger joints
- **Landmarks 13-16**: Ring finger joints
- **Landmarks 17-20**: Pinky finger joints

### Finger State Detection
For each finger, the algorithm determines if it's "up" (extended) or "down" (folded):

1. **Thumb**: Special case considering horizontal movement - compares tip position (landmark 4) with IP joint (landmark 3) and checks distance from palm center
2. **Other Fingers**: Compares tip landmarks (8, 12, 16, 20) with PIP joint landmarks (6, 10, 14, 18) on the Y-axis

### Gesture Classification Rules
- **Open Palm**: All 5 fingers extended
- **Fist**: No fingers extended (all closed)
- **Peace Sign**: Exactly 2 fingers extended (index + middle)
- **Thumbs Up**: Only thumb extended upward

### Robustness Features
- Confidence thresholds for hand detection (70%) and tracking (50%)
- Error handling for edge cases and missing landmarks
- Visual feedback with bounding boxes and gesture labels

## 📋 Requirements

- Python 3.7 or higher
- Webcam/Camera device
- Operating System: Windows, macOS, or Linux

## 🚀 Setup and Installation

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd hand-gesture-recognition
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv gesture_env

# Activate virtual environment
# On Windows:
gesture_env\Scripts\activate
# On macOS/Linux:
source gesture_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
Make sure your webcam is connected and not being used by other applications.

## 🎮 Execution Instructions

### Run the Application
```bash
python hand_gesture.py
```

### Usage
1. **Position Yourself**: Sit in front of your webcam with good lighting
2. **Show Your Hand**: Hold your hand clearly visible in the camera frame
3. **Make Gestures**: Try the supported gestures:
   - Spread all fingers for "Open Palm"
   - Close all fingers for "Fist" 
   - Show index and middle finger for "Peace Sign"
   - Show thumbs up for "Thumbs Up"
4. **Exit**: Press 'q' to quit the application

### Tips for Best Results
- Ensure good lighting conditions
- Keep your hand within the camera frame
- Avoid busy backgrounds
- Hold gestures steady for consistent recognition
- Keep reasonable distance from camera (arm's length)

## 📁 Project Structure
```
hand-gesture-recognition/
│
├── hand_gesture.py      # Main application file
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
└── .gitignore          # Git ignore file (recommended)
```

## 🔍 Troubleshooting

### Common Issues

1. **"Could not open webcam"**
   - Check if camera is connected and not used by other applications
   - Try different camera indices: `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`

2. **Poor gesture recognition**
   - Improve lighting conditions
   - Ensure hand is clearly visible and not partially occluded
   - Adjust `min_detection_confidence` and `min_tracking_confidence` values

3. **Slow performance**
   - Close other applications using camera/CPU
   - Lower the resolution in video capture settings
   - Check if hardware acceleration is available

### System Requirements
- **RAM**: Minimum 4GB recommended
- **Processor**: Modern multi-core processor
- **Camera**: 720p webcam or higher

## 🔮 Future Enhancements

- Add more gesture types (OK sign, pointing, etc.)
- Implement gesture sequence recognition
- Add gesture customization interface
- Support for multiple hands simultaneously
- Integration with system controls/automation

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---


