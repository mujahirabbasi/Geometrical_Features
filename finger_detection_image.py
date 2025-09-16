import cv2
import mediapipe as mp
import numpy as np
import sys
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configure MediaPipe Hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Landmark indices for fingers
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_TIP = 20

def get_finger_bounding_box(hand_landmarks, finger_name, base_idx, tip_idx, image_shape):
    """
    Get bounding box for finger with base at bottom edge
    """
    h, w = image_shape[:2]

    # Get base and tip points
    base_point = (
        int(hand_landmarks.landmark[base_idx].x * w),
        int(hand_landmarks.landmark[base_idx].y * h)
    )
    tip_point = (
        int(hand_landmarks.landmark[tip_idx].x * w),
        int(hand_landmarks.landmark[tip_idx].y * h)
    )

    # Calculate vector and angle
    dx = tip_point[0] - base_point[0]
    dy = tip_point[1] - base_point[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Finger length
    length = int(1.2 * np.sqrt(dx**2 + dy**2))

    # Finger width depends on finger type
    if finger_name in ["Thumb", "Pinky"]:
        width = int(length * 0.39)
    else:
        width = int(length * 0.30)

    # Calculate center point so that the base of finger aligns with bottom of bounding box
    # The center should be positioned so that the base is at the bottom edge of the box
    # Since the rotated rectangle extends length/2 in each direction from center,
    # we need to position the center at base + length/2 towards tip
    direction_x = (tip_point[0] - base_point[0]) / np.sqrt(dx**2 + dy**2) if np.sqrt(dx**2 + dy**2) > 0 else 0
    direction_y = (tip_point[1] - base_point[1]) / np.sqrt(dx**2 + dy**2) if np.sqrt(dx**2 + dy**2) > 0 else 0
    
    # Move center towards tip by length/2 so base is at bottom edge
    cx = base_point[0] + direction_x * (length / 2)
    cy = base_point[1] + direction_y * (length / 2)

    # Define rotated rectangle for bounding box
    rot_rect = ((cx, cy), (width, length), angle - 90)  
    box = cv2.boxPoints(rot_rect).astype(int)

    return box, (cx, cy, width, length, angle - 90)

def crop_finger_region(image, cx, cy, width, length, angle, finger_name="Unknown"):
    """
    Crop finger region using the rotation and cropping approach, 
    and remove black background.
    """
    h, w = image.shape[:2]
    
    # Normalize angle so finger points up
    if angle > 90 or angle < -90:
        angle += 180
    
    # Rotate image
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    
    # Crop
    x = int(cx - width/2)
    y = int(cy - length/2)
    x, y = max(0, x), max(0, y)
    x2, y2 = min(w, x + width), min(h, y + length)
    finger_crop = rotated[y:y2, x:x2]
    
    if finger_crop.size == 0:
        return finger_crop
    
    # ----------------------------
    # Remove black background only
    # ----------------------------
    # Convert to grayscale
    gray = cv2.cvtColor(finger_crop, cv2.COLOR_BGR2GRAY)
    
    # Print debugging information
    top_left_pixel = gray[0, 0]
    min_pixel_value = np.min(gray)
    print(f"Finger: {finger_name}")
    print(f"  Top-left pixel value: {top_left_pixel}")
    print(f"  Minimum pixel value: {min_pixel_value}")
    print(f"  Image shape: {gray.shape}")
    print("---")
    
    # Create mask to keep only finger content
    # Use a constant threshold value of 100
    threshold_value = 100
    
    print(f"  Using threshold value: {threshold_value}")
    
    # Create mask: pixels > threshold_value become visible, others become transparent
    _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Find the topmost non-black pixel (finger tip) and color it red
    # Convert back to BGR for processing
    finger_bgr = finger_crop.copy()
    
    # Find the topmost non-black pixel and create a larger red dot
    h, w = gray.shape
    topmost_red_pixel = None
    
    for y in range(h):
        for x in range(w):
            if mask[y, x] > 0:  # If pixel is not transparent (not black in mask)
                # Color a larger area around this pixel red
                dot_size = 5  # Radius of the red dot (11x11 total)
                for dy in range(-dot_size, dot_size + 1):
                    for dx in range(-dot_size, dot_size + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:  # Check bounds
                            finger_bgr[ny, nx] = [0, 0, 255]  # BGR format: red = (0, 0, 255)
                topmost_red_pixel = (x, y)
                break
        if topmost_red_pixel is not None:
            break
    
    if topmost_red_pixel:
        print(f"  Colored topmost pixel area at {topmost_red_pixel} red (11x11 dot)")
    
    # Create transparent background
    b, g, r = cv2.split(finger_bgr)
    rgba = cv2.merge([b, g, r, mask])  # add alpha channel
    
    return rgba   # PNG with transparency

def detect_fingers_in_image(image_path):
    """
    Detect fingers in an image and draw bounding boxes around them
    """
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return None, None
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image '{image_path}'!")
        return None, None
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect hands
    results = hands.process(image_rgb)
    
    # Create a copy for drawing
    result_image = image.copy()
    cropped_fingers = {}
    
    # Draw hand landmarks and finger bounding boxes
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks
            mp_drawing.draw_landmarks(
                result_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Process each finger
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            
            for i, finger_name in enumerate(finger_names):
                color = colors[i]

                # Get base and tip indices for each finger
                if finger_name == "Thumb":
                    base_idx, tip_idx = THUMB_CMC, THUMB_TIP
                elif finger_name == "Index":
                    base_idx, tip_idx = INDEX_MCP, INDEX_TIP
                elif finger_name == "Middle":
                    base_idx, tip_idx = MIDDLE_MCP, MIDDLE_TIP
                elif finger_name == "Ring":
                    base_idx, tip_idx = RING_MCP, RING_TIP
                elif finger_name == "Pinky":
                    base_idx, tip_idx = PINKY_MCP, PINKY_TIP

                # Get bounding box and parameters for finger
                box, (cx, cy, width, length, angle) = get_finger_bounding_box(hand_landmarks, finger_name, base_idx, tip_idx, image.shape)
                
                # Draw bounding box on result image
                cv2.polylines(result_image, [box], isClosed=True, color=color, thickness=2)
                
                # Add finger label
                cv2.putText(result_image, finger_name, (box[0][0], box[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Crop finger region
                cropped_finger = crop_finger_region(image, cx, cy, width, length, angle, finger_name)
                if cropped_finger.size > 0:
                    cropped_fingers[finger_name] = cropped_finger
    else:
        print("No hand detected in the image!")
        return None, None
    
    return result_image, cropped_fingers

def main():
    if len(sys.argv) != 2:
        print("Usage: python finger_detection_image.py <image_path>")
        print("Example: python finger_detection_image.py hand_image.jpg")
        return
    
    image_path = sys.argv[1]
    
    print(f"Processing image: {image_path}")
    
    # Create output folder
    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    # Detect fingers and draw bounding boxes
    result_image, cropped_fingers = detect_fingers_in_image(image_path)
    
    if result_image is not None and cropped_fingers:
        # Display the main result
        cv2.imshow('Finger Detection with Bounding Boxes', result_image)
        
        # Get base name for output files
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save the main result image
        main_output_path = os.path.join(output_folder, f"finger_detected_{base_name}.jpg")
        cv2.imwrite(main_output_path, result_image)
        print(f"Saved: finger_detected_{base_name}.jpg")
        
        # Save each cropped finger
        for finger_name, cropped_finger in cropped_fingers.items():
            # Save cropped finger image with transparency
            finger_output_path = os.path.join(output_folder, f"{finger_name.lower()}_{base_name}_cropped.png")
            cv2.imwrite(finger_output_path, cropped_finger)
            print(f"Saved: {finger_name.lower()}_{base_name}_cropped.png")
        
        print(f"\nAll images saved in '{output_folder}' folder")
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to process the image.")

if __name__ == "__main__":
    main()
