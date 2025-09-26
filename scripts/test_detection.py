import cv2
import face_recognition

# ========== IMAGE FACE DETECTION ==========
# Load a sample image
image_path = r"C:\Users\dhruv\Desktop\attendance_system\uploads\WIN_20250129_16_46_06_Pro.jpg"
image = face_recognition.load_image_file(image_path)

# Find faces
face_locations = face_recognition.face_locations(image, model="hog")
print(f"Found {len(face_locations)} face(s) in the photograph.")

# Convert to BGR for OpenCV
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Draw rectangles around faces
for top, right, bottom, left in face_locations:
    cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

# Show the image
cv2.imshow("Face Detection from Image", image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ========== WEBCAM FACE DETECTION ==========
# video_capture = cv2.VideoCapture(0)  # 0 = default webcam

# while True:
#     ret, frame = video_capture.read()
#     if not ret:
#         break

#     # Convert frame from BGR (OpenCV) to RGB (face_recognition)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Detect faces
#     face_locations = face_recognition.face_locations(rgb_frame, model="hog")

#     # Draw rectangles around detected faces
#     for top, right, bottom, left in face_locations:
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

#     # Display the result
#     cv2.imshow("Webcam Face Detection", frame)

#     # Press 'q' to quit
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# video_capture.release()
# cv2.destroyAllWindows()
