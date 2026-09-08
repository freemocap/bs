import cv2

# Open video
cap = cv2.VideoCapture('/Users/philipqueen/ferret_0776_P35_EO5/basler_pupil_synchronized/eye1.mp4')

# Read the first frame
ret, frame = cap.read()
if not ret:
    cap.release()
    raise Exception("Failed to read video")

# Select ROI manually
roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)

cv2.destroyWindow("Select ROI")

print(roi)

# # Loop through the video
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Crop the frame using the selected ROI
#     x, y, w, h = roi
#     cropped_frame = frame[y:y+h, x:x+w]

#     # Display the cropped frame
#     cv2.imshow('Cropped Video', cropped_frame)

#     # Break with 'q' key
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

cap.release()
cv2.destroyAllWindows()