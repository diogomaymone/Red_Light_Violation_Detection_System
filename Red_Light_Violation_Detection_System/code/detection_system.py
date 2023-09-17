# Import necessary libraries
import cv2 as cv
import numpy as np
from typing import List, Tuple
from collections import deque
from scipy.spatial import KDTree

# START OF UNIFIED INITIALIZATION------------------------------------------------------------------------------------------------------------------------------

# Initialize global variables for storing the points for each color ROI (Red, Green, Yellow)
# and for vehicle detection and direction vector
red_points = []
green_points = []
yellow_points = []
vehicle_points = []
direction_points = []

# Path to the video file
path = "videos/Traffic Lights/tl6.mp4"

# Create a new window named 'FIRST FRAME' for displaying the first video frame
cv.namedWindow('Unified Frame', cv.WINDOW_NORMAL)

# State variable to control what the click_event function is doing
# Can take the values: 'RED', 'GRN', 'YLW', 'VEHICLE', 'DIRECTION'
current_state = 'RED'  

# Function to handle mouse clicks on the OpenCV window
def unified_click_event(event, x, y, flags, param):
    global red_points, green_points, yellow_points, vehicle_points, direction_points, current_state
    
    # Check if the left button down event occurred
    if event == cv.EVENT_LBUTTONDOWN:
        points_list = None  # Initialize variable to hold the correct global list
        color = (0, 0, 0)  # Initialize color as black
        thickness = 2  # Initialize line thickness
        
        # Check the color parameter to decide which global list and color to use
        if current_state == 'RED':
            points_list = red_points
            color = (0, 0, 255)  # Red in BGR
        elif current_state == 'GRN':
            points_list = green_points
            color = (0, 255, 0)  # Green in BGR
        elif current_state == 'YLW':
            points_list = yellow_points
            color = (0, 255, 255)  # Yellow in BGR
        elif current_state == 'VEHICLE':
            points_list = vehicle_points
            color = (255, 0, 0)  # Blue in BGR
        elif current_state == 'DIRECTION':
            points_list = direction_points
            color = (0, 255, 255)  # Yellow in BGR

        # Proceed if a valid color parameter was found
        if points_list is not None:
            # Remove the oldest point if 4 points are already selected (2 for DIRECTION)
            max_points = 4 if current_state != 'DIRECTION' else 2
            if len(points_list) >= max_points:
                points_list.pop(0)
            
            # Append the new point
            points_list.append((x, y))

            # Decide the message to display based on the number of points selected
            msg = f"Select {max_points - len(points_list)} more points for {current_state} ROI" if len(points_list) < max_points else f"{current_state} ROI selected. Press any key to continue."
            
            # Create a temporary copy of the frame and annotate it
            temp_frame = first_frame.copy()
            cv.putText(temp_frame, msg, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Draw lines connecting the points
            for i, point in enumerate(points_list):
                if i > 0:
                    cv.line(temp_frame, points_list[i-1], point, color, thickness)
            
            # Close the ROI if max points are selected
            if len(points_list) == max_points:
                cv.line(temp_frame, points_list[-1], points_list[0], color, thickness)

            # Show the annotated frame
            cv.imshow('Unified Frame', temp_frame)

# Unified function to select the region of interest (ROI)
def unified_select_roi(color: str, first_frame, points_list: List[Tuple[int, int]]) -> Tuple[np.ndarray, int, int, int, int]:
    # Set the current state
    global current_state
    current_state = color
    
    # Display the initial message on the frame to ask for 4 points
    h, w, _ = first_frame.shape  # Get frame dimensions for scaling text
    font_scale = h / 600  
    text_pos = (int(w * 0.01), int(h * 0.08))
    
    msg = f"Select 4 points for {color} ROI" if color != 'DIRECTION' else "Select 2 points for DIRECTION"
    annotated_frame = first_frame.copy()
    cv.putText(annotated_frame, msg, text_pos, cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 2)
    cv.imshow('Unified Frame', annotated_frame)
    
    # Set the mouse callback function to handle ROI selection
    cv.setMouseCallback('Unified Frame', unified_click_event)
    cv.waitKey(0)  # Wait until a key is pressed

    # Create a polygon for masking the ROI based on the selected points
    polygon = np.array([points_list[-4:]], np.int32)
    xMin = min(x[0] for x in points_list)
    yMin = min(x[1] for x in points_list)
    xMax = max(x[0] for x in points_list)
    yMax = max(x[1] for x in points_list)
    # Create a sub-frame using the ROI dimensions
    frame = np.array([l[xMin:xMax] for l in first_frame[yMin:yMax]])

    return polygon, xMin, yMin, xMax, yMax, frame  # Return the polygon and coordinates for the ROI

# Initialize
try:
    cap = cv.VideoCapture(path)
    if not cap.isOpened(): 
        raise FileNotFoundError("Could not open video file.")
except Exception as e:
    print("Error:", e)
    exit()

# END OF UNIFIED INITIALIZATION------------------------------------------------------------------------------------------------------------------------------------------------



# START OF TRAFFIC LIGHT DETECTION INITIALIZATION------------------------------------------------------------------------------------------------------------------------------

_, first_frame = cap.read()

# Call the unified_select_roi function to select ROI for Red Light and store the results in respective variables
red_polygon, rxMin, ryMin, rxMax, ryMax, red_frame = unified_select_roi('RED', first_frame, red_points)

# Call the unified_select_roi function to select ROI for Green Light and store the
green_polygon, gxMin, gyMin, gxMax, gyMax, green_frame = unified_select_roi('GRN', first_frame, green_points)

# Call the select_roi function to select ROI for Yellow Light and store the results in respective variables
yellow_polygon, yxMin, yyMin, yxMax, yyMax, yellow_frame = unified_select_roi('YLW', first_frame, yellow_points)

# Call the select_roi function to select ROI for Vehicle Detection and store the results in respective variables
vehicle_polygon, vxMin, vyMin, vxMax, vyMax, vehicle_frame = unified_select_roi('VEHICLE', first_frame, vehicle_points)

# Call the select_roi function to select ROI for Direction Vector and store the results in respective variables
direction_polygon, dxMin, dyMin, dxMax, dyMax, direction_frame = unified_select_roi('DIRECTION', first_frame, direction_points)

# Close the ROI selection windows once all ROIs are selected
cv.destroyAllWindows()

# Initialize lists to store frames for each colored light (Red, Green, Yellow)
red_frames = []
green_frames = []
yellow_frames = []
vehicle_frames = []

# Read frames in a loop until the end of the video
while True:
    # Read a single frame and store the return value in 'ret'
    ret, frame = cap.read()
    
    # Check if a valid frame is read
    if ret:
        # Crop the areas corresponding to each color ROI from the read frame
        red_frame = np.array([l[rxMin:rxMax] for l in frame[ryMin:ryMax]])
        green_frame = np.array([l[gxMin:gxMax] for l in frame[gyMin:gyMax]])
        yellow_frame = np.array([l[yxMin:yxMax] for l in frame[yyMin:yyMax]])
    else:
        # Break the loop if no more frames are available
        break
    
    # Append the cropped areas to the respective lists for future analysis
    red_frames.append(red_frame)
    green_frames.append(green_frame)
    yellow_frames.append(yellow_frame)

# Reset the video to the first frame for future use if needed
cap.set(cv.CAP_PROP_POS_FRAMES, 0)

# Function that calculates the mean brightness of a frame in the V channel of the HSV color space.
def get_brightness(frame: np.ndarray) -> float:
    try:
        # Convert the color space of the frame to HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # Extract the V channel (Brightness)
        v_channel = hsv[:,:,2]
        # Compute the mean brightness of the V channel
        brightness = cv.mean(v_channel)[0]
        return brightness
    except Exception as e:
        print(f"An error occurred in get_brightness: {e}")
        return 0.0  # Return a default value

# Function to calculate mean brightness values for a sequence of frames for Red, Green, and Yellow lights.
def extract_brightness_values(red_video_frames, green_video_frames, yellow_video_frames):
    # Initialize lists to store mean brightness values for each color
    red_brightness_values = []
    green_brightness_values = []
    yellow_brightness_values = []

    # Calculate mean brightness for each frame for Red light and store in list
    for rframe in red_video_frames:
        rbrightness = get_brightness(rframe)
        red_brightness_values.append(rbrightness)
    
    # Calculate mean brightness for each frame for Green light and store in list
    for gframe in green_video_frames:
        gbrightness = get_brightness(gframe)
        green_brightness_values.append(gbrightness)
    
    # Calculate mean brightness for each frame for Yellow light and store in list
    for yframe in yellow_video_frames:
        ybrightness = get_brightness(yframe)
        yellow_brightness_values.append(ybrightness)

    return red_brightness_values, green_brightness_values, yellow_brightness_values

# Function to apply k-means clustering on a set of brightness values to find two cluster centers
def cluster_brightness_values(brightness_values: List[float]) -> Tuple[float, float]:
    try:
        # Create an Nx1 NumPy float32 array for the input to k-means clustering
        data = np.array(brightness_values, dtype=np.float32).reshape(-1, 1)
        
        # Number of clusters
        K = 2

        # Define criteria and apply k-means()
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        attempts = 10
        flags = cv.KMEANS_RANDOM_CENTERS
        
        ret, labels, centers = cv.kmeans(data, K, None, criteria, attempts, flags)
        
        # Extract the individual cluster centers
        center_1, center_2 = centers.ravel()
        
        return center_1, center_2
    except Exception as e:
        print(f"An error occurred in cluster_brightness_values: {e}")
        return 0.0, 0.0  # Return default values


# Main function to apply k-means clustering to find cluster centers for each color
def k_means(red_frames, green_frames, yellow_frames):
    # Extract mean brightness values from frames of each color
    red_brightness_values, green_brightness_values, yellow_brightness_values = extract_brightness_values(red_frames, green_frames, yellow_frames)
    
    # Find the cluster centers for each color by applying k-means on their brightness values
    rcenter_1, rcenter_2 = cluster_brightness_values(red_brightness_values)
    gcenter_1, gcenter_2 = cluster_brightness_values(green_brightness_values)
    ycenter_1, ycenter_2 = cluster_brightness_values(yellow_brightness_values)
    
    return rcenter_1, rcenter_2, gcenter_1, gcenter_2, ycenter_1, ycenter_2

# Run k-means function to get cluster centers for brightness for each color
red_center1, red_center2, green_center1, green_center2, yellow_center1, yellow_center2 = k_means(red_frames, green_frames, yellow_frames)

# Classify the centers as 'upper' or 'lower' based on their values to help in determining if a light is ON or OFF
upper_red_center, lower_red_center = max(red_center1, red_center2), min(red_center1, red_center2)
upper_green_center, lower_green_center = max(green_center1, green_center2), min(green_center1, green_center2)
upper_yellow_center, lower_yellow_center = max(yellow_center1, yellow_center2), min(yellow_center1, yellow_center2)

# Initialize the light status to 'UNKNOWN'.
which_light = "UNKNOWN"

# Initialize the previous light status to 'UNKNOWN'.
prev_light = "UNKNOWN"

# END OF TRAFFIC LIGHT DETECTION INITIALIZATION------------------------------------------------------------------------------------------------------------------------------



# START OF VEHICLE DETECTION INITIALIZATION------------------------------------------------------------------------------------------------------------------------------

# Initialize snapshot and video snippet variables
recent_detections = deque(maxlen=10)
snapshot_counter = 0
density_threshold = 17
snapshot_cooldown_counter = 0
snapshot_cooldown_threshold = 30
post_event_frames_to_capture = 0
is_video_recording = False
video_finalized = False
snapshot_pending = False
snapshot_delay_counter = 0

# Helper function to take snapshots
def take_snapshot(frame, filename):
    cv.imwrite(filename, frame)

# Helper function to calculate density of detected vectors
def calculate_density(vectors):
    return len(vectors)

# Helper function to check if two points are close
def is_close(point1, point2, threshold=50):
    distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    return distance < threshold

# Helper function to check if a detection is near recent detections
def check_near_recent_detections(recent_detections, current_vectors, distance_threshold=50):
    # Flatten the recent_detections and build a KDTree
    all_recent_points = [point for sublist in recent_detections for recent_vector in sublist for point in [recent_vector[0]]]
    if len(all_recent_points) == 0:
        return False
    kdtree = KDTree(all_recent_points)
    
    # Query the KDTree for each current point
    for current_vector in current_vectors:
        current_point = current_vector[0]
        # Query the tree for nearby points within the distance_threshold
        if len(kdtree.query_ball_point(current_point, distance_threshold)) > 0:
            return True
    return False

# Initialize background subtractor to differentiate moving objects from the background
bg_subtractor = cv.createBackgroundSubtractorMOG2(300, 400, True)

# Parameters for feature detection
feature_params = dict(maxCorners=500, qualityLevel=0.01, minDistance=10, blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Frame buffer to store 2 seconds of video frames
frame_buffer = deque(maxlen = int(cap.get(5) * 2))
FPS = cap.get(5)

# Obtain the total number of frames for loop iteration
numbFrames = int(cap.get(7))

# Crop the first frame to the user-selected ROI
vehicle_frame = np.array([l[vxMin:vxMax] for l in first_frame[vyMin:vyMax]])

# Convert the cropped frame to grayscale and detect features
prev_gray = cv.cvtColor(vehicle_frame, cv.COLOR_BGR2GRAY)
prev = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
mask = np.zeros_like(vehicle_frame)

# Warm-up phase
warm_up_frames = 20
for frameNum in range(warm_up_frames):
    ret, frame = cap.read()
    if not ret:
        print("Failed to read a warm-up frame.")
        exit()
    frame = frame[vyMin:vyMax, vxMin:vxMax]
    bg_subtractor.apply(frame)

# Initialize features for tracking after warm-up
ret, first_frame = cap.read()
if not ret:
    print("Failed to read the video.")
    exit()

run_vehicle_detection = False  # Flag to control vehicle detection

# END OF VEHICLE DETECTION INITIALIZATION------------------------------------------------------------------------------------------------------------------------------



# Main loop to continuously read and process each frame from the video.
while True:
    try:
        # Read a single frame from the VideoCapture object.
        ret, frame = cap.read()

        # Check if a frame has been successfully captured.
        if ret:
            # Crop out the ROIs for Red, Green, and Yellow lights from the frame.
            red_frame = frame[ryMin:ryMax, rxMin:rxMax]
            green_frame = frame[gyMin:gyMax, gxMin:gxMax]
            yellow_frame = frame[yyMin:yyMax, yxMin:yxMax]
            
            # Calculate the mean brightness for each of the cropped frames.
            red_brightness = get_brightness(red_frame)
            green_brightness = get_brightness(green_frame)
            yellow_brightness = get_brightness(yellow_frame)

            # Determine if Red, Green, or Yellow light is ON.
            # This is done based on the cluster centers and relative brightness levels.
            red_on = abs(red_brightness - upper_red_center) < abs(red_brightness - lower_red_center) and \
                    abs(red_brightness - upper_red_center) < abs(green_brightness - upper_green_center) and \
                    abs(red_brightness - upper_red_center) < abs(yellow_brightness - upper_yellow_center)

            green_on = abs(green_brightness - upper_green_center) < abs(green_brightness - lower_green_center) and \
                    abs(green_brightness - upper_green_center) < abs(red_brightness - upper_red_center) and \
                    abs(green_brightness - upper_green_center) < abs(yellow_brightness - upper_yellow_center)

            yellow_on = abs(yellow_brightness - upper_yellow_center) < abs(yellow_brightness - lower_yellow_center) and \
                        abs(yellow_brightness - upper_yellow_center) < abs(red_brightness - upper_red_center) and \
                        abs(yellow_brightness - upper_yellow_center) < abs(green_brightness - upper_green_center)

            # Identify which light is ON based on the boolean conditions above, and annotate it on the frame.
            if red_on and not (green_on or yellow_on):
                which_light = "RED"
            if green_on and not (red_on or yellow_on):
                which_light = "GRN"
            if yellow_on and not (red_on or green_on):
                which_light = "YLW"

            # Determine font scaling and text position based on the frame dimensions.
            h, w, _ = frame.shape
            font_scale = h / 600  # Adjust this value as needed.
            text_pos = (int(w * 0.01), int(h * 0.08))  # These values can also be adjusted as needed.

            # Add text annotation to the original frame and display it.
            cv.putText(frame, f"LIGHT: {which_light}", text_pos, cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 2)
            cv.namedWindow('Traffic Light', cv.WINDOW_NORMAL)
            cv.imshow('Traffic Light', frame)

            # Update the vehicle detection flag based on light status
            if which_light == "RED" and prev_light != "RED":
                run_vehicle_detection = True  # Start vehicle detection when light turns red
            elif which_light != "RED" and prev_light == "RED":
                run_vehicle_detection = False  # Stop vehicle detection when light is not red

            prev_light = which_light  # Update previous light status

            # Conditionally run the vehicle detection code
            if run_vehicle_detection:
                if not ret:
                    if 'out' in locals() and out.isOpened() and not video_finalized:  # Check the flag here
                        out.release()
                    break

                user_direction_vectors = []
                if snapshot_cooldown_counter > 0:
                    snapshot_cooldown_counter -= 1

                # Crop the current frame to the user-selected ROI
                vehicle_frame = frame[vyMin:vyMax, vxMin:vxMax]

                # Apply background subtraction
                fg_mask = bg_subtractor.apply(vehicle_frame)
                fg_mask_binary = np.uint8(fg_mask)
                # Use morphological operations to enhance the quality of the foreground mask
                fg_mask_binary = cv.morphologyEx(fg_mask_binary, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))

                gray = cv.cvtColor(vehicle_frame, cv.COLOR_BGR2GRAY)
                
                # Detect features and compute the optical flow for the current frame
                prev = cv.goodFeaturesToTrack(gray, mask=fg_mask_binary, **feature_params)

                # Check if any features were detected
                if prev is not None and len(prev) > 0:
                    next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
                    if next is not None:
                        good_old = prev[status == 1]
                        good_new = next[status == 1]

                        # For each detected vehicle motion, determine its direction and annotate the frame
                        for i, (new, old) in enumerate(zip(good_new, good_old)):
                            a, b = new.ravel()
                            c, d = old.ravel()

                            dx = c - a
                            dy = d - b

                            # Default motion direction, detection, indication and annotation color
                            detection = False

                            # If the user has provided a specific direction (indicated by two direction points),
                            # then compare the detected motion direction with the user's specified direction.
                            if len(direction_points) == 2:

                                # Calculate the vectors representing the user-specified direction and the detected vehicle motion direction.
                                user_arrow_vector = np.array(direction_points[1]) - np.array(direction_points[0])
                                vehicle_arrow_vector = np.array([c, d]) - np.array([a, b])

                                # Calculate the magnitudes (norms) of both vectors.
                                vehicle_norm = np.linalg.norm(vehicle_arrow_vector)
                                user_norm = np.linalg.norm(user_arrow_vector)

                                # Ensure neither vector is a zero vector (has zero magnitude) to avoid division by zero.
                                if vehicle_norm > 0 and user_norm > 0:
                                    # Calculate the cosine of the angle between the two vectors.
                                    cosine_angle = np.dot(vehicle_arrow_vector, -user_arrow_vector) / (vehicle_norm * user_norm)
                                    # Ensure the cosine value lies between -1 and 1 to avoid mathematical errors.
                                    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
                                    # Compute the angle difference using arccos.
                                    angle_diff = np.arccos(cosine_angle)

                                    # If the angle difference is less than 30 degrees (motion is similar to the user's direction),
                                    # label the detection (vehicle) as True and print a visual indication on screen with "Detected".
                                    if angle_diff < np.radians(17):
                                        detection = True

                                else:
                                    # If either vector is a zero vector, skip the current iteration.
                                    continue

                            if detection:  # 'detection' boolean is set in your existing code
                                user_direction_vectors.append((new, old))

                        # Calculate density
                        density = calculate_density(user_direction_vectors)
                        # Add the current frame to the rolling buffer
                        frame_buffer.append(frame.copy())

                        if density > density_threshold and snapshot_cooldown_counter == 0 and not is_video_recording:
                            is_near_recent = check_near_recent_detections(recent_detections, user_direction_vectors)
                            
                            if not is_near_recent:
                                # A car is detected. Initialize post-event recording.
                                post_event_frames_to_capture = FPS * 3  # 3 seconds * FPS

                                # Set the video recording flag
                                is_video_recording = True

                                video_finalized = False

                                # Create a video snippet
                                video_filename = f"video_snippet_{snapshot_counter}.mp4"
                                fourcc = cv.VideoWriter_fourcc(*'mp4v')  # FourCC code for MP4
                                out = cv.VideoWriter(video_filename, fourcc, FPS, (frame.shape[1], frame.shape[0]))

                                # Write the buffered frames (2 seconds before the event)
                                for buffered_frame in frame_buffer:
                                    out.write(buffered_frame)

                                # Set the snapshot pending flag and initialize the delay counter
                                snapshot_pending = True
                                snapshot_delay_counter = FPS // 1.5  # Delay for a second and a half 

                                # Reset the snapshot cooldown counter
                                snapshot_cooldown_counter = snapshot_cooldown_threshold
                                
                                # Add to recent detections
                                recent_detections.append(user_direction_vectors)

                        # Handle post-event recording
                        if post_event_frames_to_capture > 0:
                            if out.isOpened():  # Check the flag here
                                out.write(frame)
                            else:
                                print("Error: VideoWriter not ready.")
                                
                            post_event_frames_to_capture -= 1
                            if post_event_frames_to_capture == 0:
                                if 'out' in locals() and out.isOpened():
                                    out.release()
                                    video_finalized = True
                                else:
                                    print("Error: Could not finalize video.")
                                is_video_recording = False  # Reset the video recording flag

                        # Handle delayed snapshot
                        if snapshot_pending:
                            snapshot_delay_counter -= 1
                            if snapshot_delay_counter <= 0:
                                # Take the snapshot
                                snapshot_filename = f"snapshot_{snapshot_counter}.png"
                                take_snapshot(frame, snapshot_filename)

                                # Reset snapshot-related variables
                                snapshot_pending = False
                                snapshot_delay_counter = 0
                                snapshot_counter += 1  # Increment the snapshot counter

                    # Update the previous grayscale frame and the set of good features for the next iteration.
                    prev_gray = gray.copy()
                    prev = good_new.reshape(-1, 1, 2)

            # Exit the loop if 'q' is pressed.
            if cv.waitKey(int(21)) & 0xFF == ord('q'):
                if 'out' in locals() and out.isOpened() and not video_finalized:  # Check the flag here
                    out.release()   
                break
        else:
            # If a frame is not captured successfully, exit the loop.
            if 'out' in locals() and out.isOpened() and not video_finalized:  # Check the flag here
             out.release()
            break

    except KeyboardInterrupt:
        print("Interrupted by user. Exiting.")
        break
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Release the VideoCapture object and destroy all OpenCV windows.
cap.release()
cv.destroyAllWindows()
