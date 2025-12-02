# Install required packages
# pip install ultralytics opencv-python numpy

from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load YOLOv8 Pose model (will auto-download on first run)
model = YOLO('yolov8n-pose.pt')  # 'n' for nano, use 'yolov8m-pose.pt' or 'yolov8l-pose.pt' for higher accuracy

# Process each saved frame
frame_files = ['Frame 1.png', 'Frame 2.png', 'Frame 3.png', 
               'Frame 4.png', 'Frame 5.png', 'Frame 6.png', 'Frame 7.png']

# Create output directory
os.makedirs('pose_output', exist_ok=True)

# Store results for later metric calculation
all_keypoints = []

for frame_file in frame_files:
    # Read frame
    img = cv2.imread(frame_file)
    
    if img is None:
        print(f"Could not read {frame_file}")
        continue
    
    # Run pose estimation
    results = model(img, verbose=False)
    
    # Get keypoints (17 keypoints in COCO format)
    # Format: [x, y, confidence] for each keypoint
    keypoints = results[0].keypoints.xy.cpu().numpy()  # Shape: [num_people, 17, 2]
    keypoints_conf = results[0].keypoints.conf.cpu().numpy()  # Confidence scores
    
    if len(keypoints) > 0:
        all_keypoints.append({
            'frame': frame_file,
            'keypoints': keypoints[0],  # First person (Tiger Woods)
            'confidence': keypoints_conf[0]
        })
        
        print(f"\n{frame_file}:")
        print(f"  Detected {len(keypoints)} person(s)")
        print(f"  Average keypoint confidence: {keypoints_conf[0].mean():.3f}")
    
    # Save annotated image
    annotated_img = results[0].plot()  # Draw skeleton and keypoints
    output_path = f'pose_output/pose_{frame_file}'
    cv2.imwrite(output_path, annotated_img)
    print(f"  Saved: {output_path}")

print(f"\n✓ Processed {len(all_keypoints)} frames successfully")
print(f"✓ Pose-overlaid images saved in 'pose_output/' folder")

# Print keypoint indices for reference (COCO format)
print("\n📋 YOLOv8 Keypoint Indices (COCO format):")
keypoint_names = [
    '0: Nose', '1: Left Eye', '2: Right Eye', '3: Left Ear', '4: Right Ear',
    '5: Left Shoulder', '6: Right Shoulder', '7: Left Elbow', '8: Right Elbow',
    '9: Left Wrist', '10: Right Wrist', '11: Left Hip', '12: Right Hip',
    '13: Left Knee', '14: Right Knee', '15: Left Ankle', '16: Right Ankle'
]
for name in keypoint_names:
    print(f"  {name}")



# Function to calculate angle between three points
def calculate_angle(p1, p2, p3):
    """
    Calculate angle at point p2 formed by points p1-p2-p3
    
    Args:
        p1: First point (shoulder) - [x, y]
        p2: Middle point (elbow) - [x, y] 
        p3: Third point (wrist) - [x, y]
    
    Returns:
        Angle in degrees
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    # Create vectors from elbow to shoulder and elbow to wrist
    vector1 = p1 - p2  # shoulder to elbow
    vector2 = p3 - p2  # wrist to elbow
    
    # Calculate dot product and magnitudes
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    # Calculate cosine of angle
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    
    # Clamp to [-1, 1] to avoid numerical errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # Calculate angle in radians then convert to degrees
    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees


# Load YOLOv8-Pose model
model = YOLO('yolov8n-pose.pt')

# YOLOv8 Keypoint indices (COCO format)
# 5: Left Shoulder, 6: Right Shoulder
# 7: Left Elbow, 8: Right Elbow
# 9: Left Wrist, 10: Right Wrist
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10

# Process each frame
frame_files = ['Frame 1.png', 'Frame 2.png', 'Frame 3.png', 
               'Frame 4.png', 'Frame 5.png', 'Frame 6.png', 'Frame 7.png']

print("=" * 70)
print("ARM ANGLE ANALYSIS - Tiger Woods Golf Swing")
print("=" * 70)

results_data = []

for i, frame_file in enumerate(frame_files, 1):
    # Read frame
    img = cv2.imread(frame_file)
    
    if img is None:
        print(f"❌ Could not read {frame_file}")
        continue
    
    # Run pose estimation
    results = model(img, verbose=False)
    
    # Get keypoints
    keypoints = results[0].keypoints.xy.cpu().numpy()
    keypoints_conf = results[0].keypoints.conf.cpu().numpy()
    
    if len(keypoints) > 0:
        kp = keypoints[0]  # First person (Tiger Woods)
        conf = keypoints_conf[0]
        
        # Extract right arm keypoints
        right_shoulder = kp[RIGHT_SHOULDER]
        right_elbow = kp[RIGHT_ELBOW]
        right_wrist = kp[RIGHT_WRIST]
        
        # Extract left arm keypoints
        left_shoulder = kp[LEFT_SHOULDER]
        left_elbow = kp[LEFT_ELBOW]
        left_wrist = kp[LEFT_WRIST]
        
        # Calculate angles
        right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        # Store results
        results_data.append({
            'frame': frame_file,
            'frame_num': i,
            'right_angle': right_arm_angle,
            'left_angle': left_arm_angle,
            'right_conf': (conf[RIGHT_SHOULDER] + conf[RIGHT_ELBOW] + conf[RIGHT_WRIST]) / 3,
            'left_conf': (conf[LEFT_SHOULDER] + conf[LEFT_ELBOW] + conf[LEFT_WRIST]) / 3
        })
        
        print(f"\n📊 {frame_file} (Frame {i}):")
        print(f"   Right Arm Angle: {right_arm_angle:.1f}°")
        print(f"   Left Arm Angle:  {left_arm_angle:.1f}°")
        print(f"   Confidence: Right={conf[RIGHT_SHOULDER]:.2f}, {conf[RIGHT_ELBOW]:.2f}, {conf[RIGHT_WRIST]:.2f}")
        print(f"                Left={conf[LEFT_SHOULDER]:.2f}, {conf[LEFT_ELBOW]:.2f}, {conf[LEFT_WRIST]:.2f}")
        
        # Optional: Draw angle on image
        annotated_img = results[0].plot()
        
        # Add angle text on image
        cv2.putText(annotated_img, f"R Arm: {right_arm_angle:.1f}deg", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(annotated_img, f"L Arm: {left_arm_angle:.1f}deg", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Save annotated image
        os.makedirs('angle_output', exist_ok=True)
        cv2.imwrite(f'angle_output/angle_{frame_file}', annotated_img)

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'Frame':<15} {'Right Arm (°)':<15} {'Left Arm (°)':<15} {'Change (°)':<15}")
print("-" * 70)

for idx, data in enumerate(results_data):
    if idx == 0:
        change = "-"
    else:
        # Track right arm change (since it's the trailing arm in golf)
        change = f"+{data['right_angle'] - results_data[idx-1]['right_angle']:.1f}" if data['right_angle'] > results_data[idx-1]['right_angle'] else f"{data['right_angle'] - results_data[idx-1]['right_angle']:.1f}"
    
    print(f"{data['frame']:<15} {data['right_angle']:>12.1f}   {data['left_angle']:>12.1f}   {change:>12}")

print("=" * 70)

# Print deliverables
print("\n" + "=" * 70)
print("DELIVERABLES FOR YOUR REPORT")
print("=" * 70)

print("\n1️⃣ METRIC CHOSEN:")
print("   Arm angle (shoulder-elbow-wrist angle)")

print("\n2️⃣ FRAMES USED:")
for data in results_data:
    print(f"   - {data['frame']}")

print("\n3️⃣ NUMERICAL VALUES:")
print("   Right Arm Angles:")
for data in results_data:
    print(f"   - {data['frame']}: {data['right_angle']:.1f}°")

print("\n   Left Arm Angles:")
for data in results_data:
    print(f"   - {data['frame']}: {data['left_angle']:.1f}°")

print("\n4️⃣ CALCULATION EXPLANATION:")
print("""
   I extracted the (x, y) coordinates of the shoulder, elbow, and wrist keypoints 
   from YOLOv8-Pose detection for each frame. I then computed two vectors: one from 
   the elbow to the shoulder and another from the elbow to the wrist. Using the dot 
   product formula (cos θ = (v1 · v2) / (|v1| × |v2|)), I calculated the cosine of 
   the angle, then applied the inverse cosine (arccos) to get the angle in radians, 
   which I converted to degrees.
""")

print("\n5️⃣ GEOMETRIC INTERPRETATION:")
if len(results_data) >= 2:
    min_angle = min(results_data, key=lambda x: x['right_angle'])
    max_angle = max(results_data, key=lambda x: x['right_angle'])
    angle_range = max_angle['right_angle'] - min_angle['right_angle']
    
    print(f"""
   The right arm angle varied from {min_angle['right_angle']:.1f}° ({min_angle['frame']}) 
   to {max_angle['right_angle']:.1f}° ({max_angle['frame']}), showing a range of 
   {angle_range:.1f}°. This indicates the arm transitions from a bent position 
   (smaller angle) during the backswing to a more extended position (larger angle) 
   during follow-through, demonstrating the full extension of the arm throughout 
   the swing motion.
""")

print("\n✅ All annotated images saved in 'angle_output/' folder")
print("=" * 70)
