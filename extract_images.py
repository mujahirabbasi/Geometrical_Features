import cv2
import os
import argparse
import sys
import mediapipe as mp
import numpy as np
from pathlib import Path

def get_hand_bounding_box(frame, expand_factor=0.2):
    """
    Detect hand using MediaPipe and return expanded bounding box
    
    Args:
        frame: Input frame
        expand_factor: Factor to expand the bounding box (default: 0.2 = 20% expansion)
    
    Returns:
        tuple: (x, y, width, height) of the expanded bounding box, or None if no hand detected
    """
    # Initialize MediaPipe hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,  # Focus on one hand
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        # Get the first hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Find bounding box of hand landmarks
        x_coords = [landmark.x * width for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y * height for landmark in hand_landmarks.landmark]
        
        min_x = int(min(x_coords))
        max_x = int(max(x_coords))
        min_y = int(min(y_coords))
        max_y = int(max(y_coords))
        
        # Calculate bounding box dimensions
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        
        # Expand the bounding box
        expand_x = int(bbox_width * expand_factor)
        expand_y = int(bbox_height * expand_factor)
        
        # Calculate expanded coordinates
        expanded_min_x = max(0, min_x - expand_x)
        expanded_max_x = min(width, max_x + expand_x)
        expanded_min_y = max(0, min_y - expand_y)
        expanded_max_y = min(height, max_y + expand_y)
        
        # Return expanded bounding box
        expanded_width = expanded_max_x - expanded_min_x
        expanded_height = expanded_max_y - expanded_min_y
        
        hands.close()
        return (expanded_min_x, expanded_min_y, expanded_width, expanded_height)
    
    hands.close()
    return None

def crop_hand_region(frame, expand_factor=0.2):
    """
    Crop frame to hand region using MediaPipe detection
    
    Args:
        frame: Input frame
        expand_factor: Factor to expand the bounding box
    
    Returns:
        Cropped frame or original frame if no hand detected
    """
    bbox = get_hand_bounding_box(frame, expand_factor)
    
    if bbox is not None:
        x, y, width, height = bbox
        return frame[y:y+height, x:x+width]
    else:
        # If no hand detected, return center crop as fallback
        h, w = frame.shape[:2]
        crop_ratio = 0.8
        new_h = int(h * crop_ratio)
        new_w = int(w * crop_ratio)
        start_y = (h - new_h) // 2
        start_x = (w - new_w) // 2
        return frame[start_y:start_y+new_h, start_x:start_x+new_w]

def extract_images_from_video(video_path, output_dir, num_images=30):
    """
    Extract evenly distributed images from a video file
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save extracted images
        num_images (int): Number of images to extract (default: 30)
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Extracting {num_images} images...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate frame intervals
    if total_frames <= num_images:
        # If video has fewer frames than requested images, extract all frames
        frame_indices = list(range(total_frames))
        print(f"Video has only {total_frames} frames, extracting all frames")
    else:
        # Calculate evenly distributed frame indices
        frame_indices = []
        step = total_frames / num_images
        for i in range(num_images):
            frame_index = int(i * step)
            frame_indices.append(frame_index)
    
    extracted_count = 0
    
    for i, frame_index in enumerate(frame_indices):
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_index}")
            continue
        
        # Generate filename
        timestamp = frame_index / fps if fps > 0 else frame_index
        filename = f"frame_{i+1:03d}_time_{timestamp:.2f}s.jpg"
        output_path = os.path.join(output_dir, filename)
        
        # Save frame
        success = cv2.imwrite(output_path, frame)
        if success:
            extracted_count += 1
            print(f"Extracted: {filename}")
        else:
            print(f"Error: Could not save frame {filename}")
    
    cap.release()
    print(f"Successfully extracted {extracted_count} images to: {output_dir}")
    return True

def extract_images_from_video_with_prefix(video_path, output_dir, prefix_name, num_images=30, rotate_90=True, crop_hand=True, expand_factor=0.2):
    """
    Extract evenly distributed images from a video file with a prefix name
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save extracted images
        prefix_name (str): Prefix for the image filenames (e.g., "Mujahir")
        num_images (int): Number of images to extract (default: 30)
        rotate_90 (bool): Whether to rotate the frame by 90 degrees (default: True)
        crop_hand (bool): Whether to crop around detected hand (default: True)
        expand_factor (float): Factor to expand hand bounding box (default: 0.2)
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Extracting {num_images} images with prefix: {prefix_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate frame intervals
    if total_frames <= num_images:
        # If video has fewer frames than requested images, extract all frames
        frame_indices = list(range(total_frames))
        print(f"Video has only {total_frames} frames, extracting all frames")
    else:
        # Calculate evenly distributed frame indices
        frame_indices = []
        step = total_frames / num_images
        for i in range(num_images):
            frame_index = int(i * step)
            frame_indices.append(frame_index)
    
    extracted_count = 0
    
    for i, frame_index in enumerate(frame_indices):
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_index}")
            continue
        
        # Rotate frame by 90 degrees if requested
        if rotate_90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # Crop around detected hand if requested
        if crop_hand:
            frame = crop_hand_region(frame, expand_factor)
        
        # Generate filename with prefix
        filename = f"{prefix_name}_frame{i+1:03d}.jpg"
        output_path = os.path.join(output_dir, filename)
        
        # Save frame
        success = cv2.imwrite(output_path, frame)
        if success:
            extracted_count += 1
            print(f"Extracted: {filename}")
        else:
            print(f"Error: Could not save frame {filename}")
    
    cap.release()
    print(f"Successfully extracted {extracted_count} images to: {output_dir}")
    return True

def process_videos_folder(videos_dir="videos", images_dir="images", num_images=30, no_rotate=False, no_crop=False, expand_factor=0.2):
    """
    Process all videos in the videos folder and extract images with folder name prefix
    
    Args:
        videos_dir (str): Path to videos directory
        images_dir (str): Path to images directory (will be created)
        num_images (int): Number of images to extract per video
    """
    if not os.path.exists(videos_dir):
        print(f"Error: Videos directory '{videos_dir}' does not exist")
        return
    
    # Create images directory
    os.makedirs(images_dir, exist_ok=True)
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    processed_count = 0
    
    for root, dirs, files in os.walk(videos_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_path = os.path.join(root, file)
                
                # Calculate relative path from videos directory
                rel_path = os.path.relpath(root, videos_dir)
                
                # Get folder name for prefix (e.g., "Mujahir" from "videos/Mujahir/")
                if rel_path == ".":
                    # Video is directly in videos folder
                    folder_name = os.path.splitext(file)[0]
                    output_subdir = os.path.join(images_dir, folder_name)
                else:
                    # Video is in a subfolder - use the folder name
                    folder_name = rel_path
                    output_subdir = os.path.join(images_dir, rel_path)
                
                print(f"\n{'='*60}")
                print(f"Processing: {video_path}")
                print(f"Folder name: {folder_name}")
                print(f"Output directory: {output_subdir}")
                print(f"{'='*60}")
                
                # Extract images with folder name prefix
                if extract_images_from_video_with_prefix(video_path, output_subdir, folder_name, num_images, 
                                                       rotate_90=not no_rotate, crop_hand=not no_crop, expand_factor=expand_factor):
                    processed_count += 1
                else:
                    print(f"Failed to process: {video_path}")
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed {processed_count} video(s)")
    print(f"Images saved in: {images_dir}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description='Extract evenly distributed images from video files')
    parser.add_argument('--video', '-v', type=str, help='Path to a single video file')
    parser.add_argument('--output', '-o', type=str, help='Output directory for images')
    parser.add_argument('--num-images', '-n', type=int, default=30, help='Number of images to extract (default: 30)')
    parser.add_argument('--videos-dir', type=str, default='videos', help='Videos directory to process (default: videos)')
    parser.add_argument('--images-dir', type=str, default='images', help='Images directory (default: images)')
    parser.add_argument('--process-all', '-a', action='store_true', help='Process all videos in videos directory')
    parser.add_argument('--no-rotate', action='store_true', help='Disable 90-degree rotation (default: rotate enabled)')
    parser.add_argument('--no-crop', action='store_true', help='Disable hand-based cropping (default: crop enabled)')
    parser.add_argument('--expand-factor', type=float, default=0.2, help='Factor to expand hand bounding box (default: 0.2)')
    
    args = parser.parse_args()
    
    try:
        if args.video:
            # Process single video
            if not args.output:
                # Create output directory
                args.output = 'images'
            
            # Get prefix name from video path
            video_dir = os.path.dirname(args.video)
            if 'videos' in video_dir:
                # Extract folder name from videos/Mujahir/video1.mp4 -> Mujahir
                rel_path = os.path.relpath(video_dir, 'videos')
                if rel_path == ".":
                    prefix_name = os.path.splitext(os.path.basename(args.video))[0]
                    output_subdir = os.path.join(args.output, prefix_name)
                else:
                    prefix_name = rel_path
                    output_subdir = os.path.join(args.output, rel_path)
            else:
                prefix_name = os.path.splitext(os.path.basename(args.video))[0]
                output_subdir = os.path.join(args.output, prefix_name)
            
            print(f"Extracting {args.num_images} images from: {args.video}")
            print(f"Output directory: {output_subdir}")
            print(f"Prefix name: {prefix_name}")
            
            if extract_images_from_video_with_prefix(args.video, output_subdir, prefix_name, args.num_images, 
                                                   rotate_90=not args.no_rotate, crop_hand=not args.no_crop, expand_factor=args.expand_factor):
                print("Image extraction completed successfully!")
            else:
                print("Image extraction failed!")
                sys.exit(1)
                
        elif args.process_all:
            # Process all videos in videos directory
            print(f"Processing all videos in: {args.videos_dir}")
            print(f"Images will be saved in: {args.images_dir}")
            print(f"Extracting {args.num_images} images per video")
            
            process_videos_folder(args.videos_dir, args.images_dir, args.num_images, args.no_rotate, args.no_crop, args.expand_factor)
            
        else:
            # Default: process all videos in videos directory
            print("No specific video provided. Processing all videos in 'videos' directory...")
            process_videos_folder(args.videos_dir, args.images_dir, args.num_images, args.no_rotate, args.no_crop, args.expand_factor)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
