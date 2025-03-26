import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

# Parameters
buffer_size = 300  # Number of points to keep in the trail
midpoint_lower = np.array([30, 100, 50])  # HSV lower bounds for green
midpoint_upper = np.array([80, 255, 255])  # HSV upper bounds for green

bottom_lower = np.array([140, 100, 100])  # HSV lower bounds for green
bottom_upper = np.array([180, 255, 255])  # HSV upper bounds for green
fade_factor = 0.9  # How quickly points fade (smaller = faster fade)

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for default camera, change if needed

# Create a deque to store the path points with their intensities
midpoint_pts = deque(maxlen=buffer_size)
midpoint_intensities = deque(maxlen=buffer_size)

bottom_pts = deque(maxlen=buffer_size)
bottom_intensities = deque(maxlen=buffer_size)

# Create a plotting figure
plt.figure(figsize=(10, 8))
plt.xlim(0, 640)  # Adjust to your camera resolution
plt.ylim(0, 480)
plt.gca().invert_yaxis()  # Invert y-axis to match camera coordinates
path_plot, = plt.plot([], [], 'r-', linewidth=2)
plt.grid(True)
plt.title("Double Pendulum Path")
plt.xlabel("X position")
plt.ylabel("Y position")

# Function to update plot
def update_plot():
    x_points = [pt[0] for pt in midpoint_pts]
    y_points = [pt[1] for pt in midpoint_pts]
    colors = []
    
    bottom_x_points = [pt[0] for pt in bottom_pts]
    bottom_y_points = [pt[1] for pt in bottom_pts]
    bottom_colors = []
    
    # Create color array based on intensities
    for i in bottom_intensities:
        bottom_colors.append((1, 0, 0, i))  # RGBA red with varying alpha

    
    # Create color array based on intensities
    for i in midpoint_intensities:
        colors.append((0, 1, 0, i))  # RGBA red with varying alpha
    
    plt.clf()
    plt.xlim(0, 640)
    plt.ylim(0, 480)
    plt.gca().invert_yaxis()
    
    # Plot points with varying alpha
    for i in range(1, len(midpoint_pts)):
        plt.plot([x_points[i-1], x_points[i]], [y_points[i-1], y_points[i]], 
                    color=colors[i], linewidth=2, label="Midpoint")
    
    for i in range(1, len(bottom_pts)):
        plt.plot([bottom_x_points[i-1], bottom_x_points[i]], [bottom_y_points[i-1], bottom_y_points[i]], 
                    color=bottom_colors[i], linewidth=2, label="Bottom")
    
    plt.grid(True)
    plt.title("Double Pendulum Path")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.pause(0.001)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to HSV and find the marker
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    midpoint_mask = cv2.inRange(hsv, midpoint_lower, midpoint_upper)
    midpoint_mask = cv2.erode(midpoint_mask, None, iterations=2)
    midpoint_mask = cv2.dilate(midpoint_mask, None, iterations=2)
    
    # Find contours
    midpoint_contours, _ = cv2.findContours(midpoint_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Only proceed if at least one contour was found
    if len(midpoint_contours) > 0:
        # Find the largest contour
        c = max(midpoint_contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        # Only proceed if the radius meets a minimum size
        if radius > 3:
            # Draw the circle on the frame
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            
            # Update points list
            midpoint_pts.appendleft(center)
            midpoint_intensities.appendleft(1.0)  # New point has full intensity
            
            # Reduce intensity of all previous points
    midpoint_intensities = deque([i * fade_factor for i in midpoint_intensities], maxlen=buffer_size)
    
    
    bottom_mask = cv2.inRange(hsv, bottom_lower, bottom_upper)
    bottom_mask = cv2.erode(bottom_mask, None, iterations=2)
    bottom_mask = cv2.dilate(bottom_mask, None, iterations=2)
    # Find contours
    bottom_contours, _ = cv2.findContours(bottom_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Only proceed if at least one contour was found
    if len(bottom_contours) > 0:
        # Find the largest contour
        c = max(bottom_contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        # Only proceed if the radius meets a minimum size
        if radius > 3:
            # Draw the circle on the frame
            cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)
            
            # Update points list
            bottom_pts.appendleft(center)
            bottom_intensities.appendleft(1.0)  # New point has full intensity
            
            # Reduce intensity of all previous points
    bottom_intensities = deque([i * fade_factor for i in bottom_intensities], maxlen=buffer_size)
    
    # Draw the path
    for i in range(1, len(midpoint_pts)):
        thickness = int(np.sqrt(buffer_size / float(i + 1)) * 2)
        alpha = midpoint_intensities[i]
        color = (0, 255, 0, int(alpha * 255))  # RGBA color with alpha
        cv2.line(frame, midpoint_pts[i-1], midpoint_pts[i], (0, 255, 0), thickness)
    
    for i in range(1, len(bottom_pts)):
        thickness = int(np.sqrt(buffer_size / float(i + 1)) * 2.5)
        alpha = bottom_intensities[i]
        color = (0, 0, 255, int(alpha * 255))  # RGBA color with alpha
        cv2.line(frame, bottom_pts[i-1], bottom_pts[i], (0, 0, 255), thickness)
    
    # Show the frame
    cv2.imshow("Frame", frame)
    
    # Update the plot
    update_plot()
    
    # Exit if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release everything when done
cap.release()
cv2.destroyAllWindows()
plt.close()