#!/usr/bin/env python3
"""
Hand Gesture Recognition System
Real-time hand gesture recognition using MediaPipe and OpenCV
Recognizes: Open Palm, Fist, Peace Sign, and Thumbs Up
"""

import cv2
import mediapipe as mp
import numpy as np

class HandGestureRecognizer:
    def __init__(self):
        """Initialize MediaPipe hands and drawing utilities"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def get_landmark_positions(self, hand_landmarks, frame_shape):
        """Extract landmark positions from hand landmarks"""
        h, w, c = frame_shape
        landmark_list = []
        
        for id, lm in enumerate(hand_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmark_list.append([cx, cy])
            
        return landmark_list
    
    def is_finger_up(self, landmarks, finger_tip, finger_pip):
        """Check if a finger is extended (up)"""
        return landmarks[finger_tip][1] < landmarks[finger_pip][1]
    
    def is_thumb_up(self, landmarks):
        """Check if thumb is extended (considering handedness)"""
        # Thumb tip (4) should be higher than thumb IP (3)
        # and thumb should be away from palm center
        thumb_up = landmarks[4][1] < landmarks[3][1]
        thumb_away = abs(landmarks[4][0] - landmarks[9][0]) > abs(landmarks[3][0] - landmarks[9][0])
        return thumb_up and thumb_away
    
    def recognize_gesture(self, landmarks):
        """
        Recognize hand gesture based on landmark positions
        Returns gesture name as string
        """
        if not landmarks or len(landmarks) < 21:
            return "No Hand Detected"
        
        # Check which fingers are up
        fingers_up = []
        
        # Thumb (special case - horizontal movement)
        fingers_up.append(self.is_thumb_up(landmarks))
        
        # Other fingers (8, 12, 16, 20 are tips; 6, 10, 14, 18 are PIPs)
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            fingers_up.append(self.is_finger_up(landmarks, tip, pip))
        
        # Count total fingers up
        total_fingers = sum(fingers_up)
        
        # Gesture recognition logic
        if total_fingers == 5:
            return "Open Palm"
        elif total_fingers == 0:
            return "Fist"
        elif total_fingers == 2 and fingers_up[1] and fingers_up[2]:  # Index and middle
            return "Peace Sign"
        elif total_fingers == 1 and fingers_up[0]:  # Only thumb
            return "Thumbs Up"
        elif total_fingers == 1 and fingers_up[1]:  # Only index
            return "Pointing Up"
        elif total_fingers == 1 and fingers_up[2]:  # Only middle
            return "Middle Finger"
        elif total_fingers == 1 and fingers_up[3]:  # Only ring
            return "Ring Finger"
        elif total_fingers == 1 and fingers_up[4]:  # Only pinky
            return "Pinky Up"
        else:
            return f"Unknown ({total_fingers} fingers)"
    
    def draw_gesture_info(self, frame, gesture, landmarks):
        """Draw gesture information on the frame"""
        if landmarks:
            # Draw gesture text near the wrist
            text_x = max(landmarks[0][0] - 100, 10)
            text_y = max(landmarks[0][1] - 50, 50)
            
            # Draw background rectangle for better text visibility
            text_size = cv2.getTextSize(gesture, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cv2.rectangle(frame, 
                         (text_x - 10, text_y - text_size[1] - 10),
                         (text_x + text_size[0] + 10, text_y + 10),
                         (0, 0, 0), -1)
            
            # Draw gesture text
            cv2.putText(frame, gesture, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            # Draw "No Hand Detected" at top-left
            cv2.putText(frame, gesture, (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    def run(self):
        """Main application loop"""
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Hand Gesture Recognition Started")
        print("Press 'q' to quit")
        print("Supported gestures: Open Palm, Fist, Peace Sign, Thumbs Up")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.hands.process(frame_rgb)
            
            gesture = "No Hand Detected"
            landmarks = None
            
            # Process hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Get landmark positions
                    landmarks = self.get_landmark_positions(hand_landmarks, frame.shape)
                    
                    # Recognize gesture
                    gesture = self.recognize_gesture(landmarks)
                    
                    # Only process first hand
                    break
            
            # Draw gesture information
            self.draw_gesture_info(frame, gesture, landmarks)
            
            # Add instructions
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Display the frame
            cv2.imshow('Hand Gesture Recognition', frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed")

def main():
    """Main function to run the hand gesture recognizer"""
    try:
        recognizer = HandGestureRecognizer()
        recognizer.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()