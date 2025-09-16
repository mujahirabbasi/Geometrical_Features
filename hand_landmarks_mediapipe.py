import cv2
import mediapipe as mp
import argparse
import sys
import csv
import numpy as np
import math
from datetime import datetime

class HandLandmarkDetector:
    def __init__(self, class_label=None, csv_filename=None):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # CSV logging setup
        self.class_label = class_label
        self.csv_filename = csv_filename or f"{class_label}.csv" if class_label else "hand_features.csv"
        self.csv_data = []
        
        # Initialize CSV file with headers
        if self.class_label:
            self.init_csv()
    
    def init_csv(self):
        """Initialize CSV file with headers"""
        headers = [
            'class_label',
            'thumb_relative_length', 'index_relative_length', 'ring_relative_length', 'pinky_relative_length',
            'thumb_curvature', 'index_curvature', 'middle_curvature', 'ring_curvature', 'pinky_curvature',
            'thumb_aspect_ratio', 'index_aspect_ratio', 'middle_aspect_ratio', 'ring_aspect_ratio', 'pinky_aspect_ratio',
            'thumb_section_diff', 'index_section_diff', 'middle_section_diff', 'ring_section_diff', 'pinky_section_diff'
        ]
        
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two 3D points"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)
    
    def calculate_curvature(self, points):
        """Calculate curvature of a finger using three points (angle-based)"""
        if len(points) < 3:
            return 0
        
        # Use three key points: tip, middle joint, base
        p1, p2, p3 = points[0], points[1], points[2]
        
        # Calculate vectors
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        
        # Calculate angle between vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)  # Avoid numerical errors
        angle = math.acos(cos_angle)
        
        return math.degrees(angle)
    
    def calculate_aspect_ratio(self, points):
        """Calculate aspect ratio (length/width) of a finger"""
        if len(points) < 4:
            return 0
        
        # Calculate finger length (tip to base)
        length = self.calculate_distance(points[0], points[-1])
        
        # Calculate finger width (average of joint widths)
        width_sum = 0
        for i in range(1, len(points)-1):
            # Calculate perpendicular distance from joint to main axis
            width = abs((points[i].y - points[0].y) * (points[-1].x - points[0].x) - 
                       (points[i].x - points[0].x) * (points[-1].y - points[0].y)) / length
            width_sum += width
        
        avg_width = width_sum / (len(points) - 2) if len(points) > 2 else 1
        
        return length / avg_width if avg_width > 0 else 0
    
    def calculate_section_differences(self, points):
        """Calculate differences between finger sections"""
        if len(points) < 4:
            return 0
        
        sections = []
        for i in range(len(points) - 1):
            section_length = self.calculate_distance(points[i], points[i+1])
            sections.append(section_length)
        
        if len(sections) < 2:
            return 0
        
        # Calculate variance in section lengths
        mean_length = sum(sections) / len(sections)
        variance = sum((length - mean_length)**2 for length in sections) / len(sections)
        
        return math.sqrt(variance)
    
    def extract_finger_features(self, landmarks):
        """Extract features for all fingers"""
        # Define finger landmark indices (MediaPipe hand landmarks)
        finger_indices = {
            'thumb': [4, 3, 2, 1],         # Thumb: tip to base (using point 1 as base)
            'index': [8, 7, 6, 5],         # Index finger
            'middle': [12, 11, 10, 9],     # Middle finger
            'ring': [16, 15, 14, 13],      # Ring finger
            'pinky': [20, 19, 18, 17]      # Pinky finger
        }
        
        features = {}
        
        # Get finger points
        finger_points = {}
        for finger_name, indices in finger_indices.items():
            finger_points[finger_name] = [landmarks.landmark[i] for i in indices]
        
        # Calculate middle finger length as reference
        middle_length = self.calculate_distance(finger_points['middle'][0], finger_points['middle'][-1])
        
        # Extract features for each finger
        for finger_name, points in finger_points.items():
            # Relative length to middle finger
            finger_length = self.calculate_distance(points[0], points[-1])
            relative_length = (finger_length / middle_length * 100) if middle_length > 0 else 0
            features[f'{finger_name}_relative_length'] = relative_length
            
            # Curvature
            curvature = self.calculate_curvature(points[:3])  # Use first 3 points
            features[f'{finger_name}_curvature'] = curvature
            
            # Aspect ratio
            aspect_ratio = self.calculate_aspect_ratio(points)
            features[f'{finger_name}_aspect_ratio'] = aspect_ratio
            
            # Section differences
            section_diff = self.calculate_section_differences(points)
            features[f'{finger_name}_section_diff'] = section_diff
        
        return features

    def save_features_to_csv(self, features):
        """Save extracted features to CSV file"""
        if not self.class_label:
            return
        
        row = [self.class_label]
        row.extend([
            round(features.get('thumb_relative_length', 0), 2),
            round(features.get('index_relative_length', 0), 2),
            round(features.get('ring_relative_length', 0), 2),
            round(features.get('pinky_relative_length', 0), 2),
            round(features.get('thumb_curvature', 0), 2),
            round(features.get('index_curvature', 0), 2),
            round(features.get('middle_curvature', 0), 2),
            round(features.get('ring_curvature', 0), 2),
            round(features.get('pinky_curvature', 0), 2),
            round(features.get('thumb_aspect_ratio', 0), 2),
            round(features.get('index_aspect_ratio', 0), 2),
            round(features.get('middle_aspect_ratio', 0), 2),
            round(features.get('ring_aspect_ratio', 0), 2),
            round(features.get('pinky_aspect_ratio', 0), 2),
            round(features.get('thumb_section_diff', 0), 2),
            round(features.get('index_section_diff', 0), 2),
            round(features.get('middle_section_diff', 0), 2),
            round(features.get('ring_section_diff', 0), 2),
            round(features.get('pinky_section_diff', 0), 2)
        ])
        
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

    def process_frame(self, frame, save_features=False, rotate_90=False):
        """Process a single frame and draw hand landmarks"""
        # Rotate frame by 90 degrees if requested
        if rotate_90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks and extract features
        if results.multi_hand_landmarks:
            # Process only the first detected hand to avoid multiple rows per image
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks and connections
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Extract and save features only when requested
            if self.class_label and save_features:
                features = self.extract_finger_features(hand_landmarks)
                self.save_features_to_csv(features)
                print(f"Features saved for class: {self.class_label}")
        
        return frame

    def run_webcam(self):
        """Run hand landmark detection on webcam feed"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Print instructions
        if self.class_label:
            print("Press 'q' to quit, 's' to save features")
        else:
            print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                break
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            save_features = False
            
            if key == ord('q'):
                break
            elif key == ord('s') and self.class_label:
                save_features = True
            
            # Process frame
            processed_frame = self.process_frame(frame, save_features)
            
            # Display frame
            cv2.imshow('Hand Landmarks - Webcam', processed_frame)
        
        cap.release()
        cv2.destroyAllWindows()

    def run_video(self, video_path):
        """Run hand landmark detection on video file"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_delay = int(1000 / fps) if fps > 0 else 30
        
        # Print instructions
        if self.class_label:
            print(f"Press 'q' to quit, 'p' to pause/resume, 's' to save features")
        else:
            print(f"Press 'q' to quit, 'p' to pause/resume")
        print(f"Video FPS: {fps}")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached")
                    break
                
                # Check for key presses
                key = cv2.waitKey(frame_delay) & 0xFF
                save_features = False
                
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key == ord('s') and self.class_label:
                    save_features = True
                
                # Process frame with 90-degree rotation for video
                processed_frame = self.process_frame(frame, save_features, rotate_90=True)
                
                # Display frame
                cv2.imshow('Hand Landmarks - Video', processed_frame)
            else:
                # Handle key presses when paused
                key = cv2.waitKey(frame_delay) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
        
        cap.release()
        cv2.destroyAllWindows()

    def run_images_folder(self, images_folder):
        """Run hand landmark detection on all images in a folder"""
        import os
        import glob
        import time
        
        # Start timing
        start_time = time.time()
        
        # Supported image extensions
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        
        # Find all image files in the folder
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(images_folder, ext)))
            image_files.extend(glob.glob(os.path.join(images_folder, ext.upper())))
        
        if not image_files:
            print(f"Error: No image files found in folder: {images_folder}")
            return
        
        # Remove duplicates and sort image files for consistent processing order
        image_files = list(set(image_files))  # Remove duplicates
        image_files.sort()
        
        print(f"Found {len(image_files)} images in folder: {images_folder}")
        
        if self.class_label:
            print("Processing all images and saving features automatically...")
        else:
            print("Processing all images...")
        
        processed_count = 0
        features_saved_count = 0
        
        for i, image_path in enumerate(image_files):
            print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # Read image
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Warning: Could not read image: {image_path}")
                continue
            
            # Process frame without rotation (images are already rotated from extraction)
            processed_frame = self.process_frame(frame, save_features=self.class_label is not None, rotate_90=False)
            
            # Display frame
            cv2.imshow('Hand Landmarks - Images', processed_frame)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Processing stopped by user")
                break
            
            processed_count += 1
            
            # Count features saved
            if self.class_label:
                features_saved_count += 1
        
        cv2.destroyAllWindows()
        
        # Calculate processing time
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\nProcessing complete!")
        print(f"Images processed: {processed_count}")
        if self.class_label:
            print(f"Features saved: {features_saved_count}")
            print(f"CSV file: {self.csv_filename}")
        
        # Display timing information
        print(f"\n⏱️  Timing Information:")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per image: {total_time/processed_count:.2f} seconds")
        print(f"Processing speed: {processed_count/total_time:.1f} images/second")

def main():
    parser = argparse.ArgumentParser(description='Hand Landmark Detection using MediaPipe with Feature Extraction')
    parser.add_argument('--video', '-v', type=str, help='Path to video file')
    parser.add_argument('--images', '-i', type=str, help='Path to images folder')
    parser.add_argument('--webcam', '-w', action='store_true', help='Use webcam feed')
    parser.add_argument('--class', '-c', type=str, dest='class_label', help='Class label for data collection (e.g., "open_hand", "fist", "point")')
    parser.add_argument('--csv', type=str, help='Custom CSV filename (optional)')
    
    args = parser.parse_args()
    
    # Create detector instance with class label
    detector = HandLandmarkDetector(class_label=args.class_label, csv_filename=args.csv)
    
    # Print information about data collection
    if args.class_label:
        print(f"Data collection mode: ON")
        print(f"Class label: {args.class_label}")
        print(f"CSV file: {detector.csv_filename}")
        print("Features being collected:")
        print("- Finger relative lengths (compared to middle finger)")
        print("- Finger curvatures")
        print("- Finger aspect ratios")
        print("- Finger section differences")
        print("Press 's' to save features for current frame")
        print("Press 'q' to quit")
    else:
        print("Data collection mode: OFF (use --class to enable)")
    
    try:
        if args.video:
            print(f"Processing video: {args.video}")
            detector.run_video(args.video)
        elif args.images:
            print(f"Processing images folder: {args.images}")
            detector.run_images_folder(args.images)
        elif args.webcam:
            print("Starting webcam feed...")
            detector.run_webcam()
        else:
            # Default to webcam if no arguments provided
            print("Starting webcam feed (default)...")
            detector.run_webcam()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
