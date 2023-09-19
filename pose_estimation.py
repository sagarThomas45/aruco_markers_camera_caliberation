import cv2
import numpy as np

# Define constants
dictionaryId = cv2.aruco.DICT_4X4_50
marker_length_m = 0.1  # Adjust this based on your marker's actual size
wait_time = 10

# Initialize the camera
videoInput = 0  # You can change this to your camera source if needed
cap = cv2.VideoCapture(videoInput)

if not cap.isOpened():
    print(f"Failed to open video input: {videoInput}")
    exit(1)

# Load the ArUco dictionary
aruco_dict = cv2.aruco.Dictionary_get(dictionaryId)

# Load camera parameters (camera_matrix and dist_coeffs) from a calibration file
fs = cv2.FileStorage("calibration_params.yml", cv2.FILE_STORAGE_READ)
camera_matrix = fs.getNode("camera_matrix").mat()
dist_coeffs = fs.getNode("distortion_coefficients").mat()

while True:
    ret, image = cap.read()
    if not ret:
        break

    image_copy = image.copy()
    ids = []
    corners = []

    # Detect ArUco markers in the image
    corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict)

    if ids:
        cv2.aruco.drawDetectedMarkers(image_copy, corners, ids)

        # Estimate pose for each detected marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length_m, camera_matrix, dist_coeffs)

        for i in range(len(ids)):
            cv2.aruco.drawAxis(image_copy, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)

            # Print the pose information for each marker
            translation = tvecs[i][0]
            rotation = rvecs[i][0]
            print(f"Marker ID {ids[i][0]} - Translation: {translation}, Rotation: {rotation}")

    cv2.imshow("Pose Estimation", image_copy)
    key = cv2.waitKey(wait_time)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
