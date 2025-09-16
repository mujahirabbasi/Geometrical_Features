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

def crop_and_rotate_finger(image, hand_landmarks, finger_name, base_idx, tip_idx):
    """
    Crop finger region and rotate it to be 90 degrees to x-axis (vertical)
    """
    h, w = image.shape[:2]
    
    # Get base and tip points
    base_point = (
        int(hand_landmarks.landmark[base_idx].x * w),
        int(hand_landmarks.landmark[base_idx].y * h)
    )
    tip_point = (
        int(hand_landmarks.landmark[tip_idx].x * w),
        int(hand_landmarks.landmark[tip_idx].y * h)
    )

    # Calculate direction vector and angle
    dx = tip_point[0] - base_point[0]
    dy = tip_point[1] - base_point[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Calculate extended tip point (1.2x extension towards tip)
    extended_tip_x = base_point[0] + int(1.2 * dx)
    extended_tip_y = base_point[1] + int(1.2 * dy)
    extended_tip = (extended_tip_x, extended_tip_y)

    # Center of finger (between base and extended tip)
    cx = int((base_point[0] + extended_tip[0]) / 2)
    cy = int((base_point[1] + extended_tip[1]) / 2)

    # Finger length and width (with padding)
    length = int(1.2 * np.sqrt(dx**2 + dy**2))   # extended length from base to extended tip
    if finger_name == "Thumb" or finger_name == "Pinky":
        width = int(length * 0.39)               # keep thumb and pinky width as is
    else:
        width = int(length * 0.30)               # set three middle fingers to 0.30

    # Create rotated rectangle for cropping
    rot_rect = ((cx, cy), (width, length), angle - 90)
    box = cv2.boxPoints(rot_rect)
    box = np.int0(box)
    
    # Get bounding rectangle for cropping
    x, y, w_crop, h_crop = cv2.boundingRect(box)
    
    # Add padding
    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w_crop = min(w - x, w_crop + 2 * padding)
    h_crop = min(h - y, h_crop + 2 * padding)
    
    # Crop the region
    cropped = image[y:y+h_crop, x:x+w_crop]
    
    if cropped.size == 0:
        return None, None
    
    # Calculate rotation angle to make finger vertical (90 degrees to x-axis)
    rotation_angle = 90 - angle
    
    # Get rotation matrix
    center = (w_crop // 2, h_crop // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    
    # Rotate the cropped image
    rotated = cv2.warpAffine(cropped, rotation_matrix, (w_crop, h_crop), 
                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated, box

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

                # Crop and rotate finger
                rotated_finger, box = crop_and_rotate_finger(image, hand_landmarks, finger_name, base_idx, tip_idx)
                
                if rotated_finger is not None:
                    cropped_fingers[finger_name] = rotated_finger
                    
                    # Draw bounding box on result image
                    cv2.polylines(result_image, [box], isClosed=True, color=color, thickness=2)
                    
                    # Add finger label
                    cv2.putText(result_image, finger_name, (box[0][0], box[0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
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
    
    # Detect fingers and draw bounding boxes
    result_image, cropped_fingers = detect_fingers_in_image(image_path)
    
    if result_image is not None and cropped_fingers:
        # Display the main result
        cv2.imshow('Finger Detection with Bounding Boxes', result_image)
        
        # Display each cropped and rotated finger
        for finger_name, cropped_finger in cropped_fingers.items():
            cv2.imshow(f'{finger_name} Finger (Cropped & Rotated)', cropped_finger)
        
        print("Press any key to close all windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to process the image.")

if __name__ == "__main__":
    main()
