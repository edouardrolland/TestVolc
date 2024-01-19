# Open the video file
video_path = r"C:\Users\edoua\Documents\Birse\Bristol\MSc Thesis\TEST_1\full_flight_640.mp4"
import cv2
from ultralytics import YOLO
import supervision as sv


image_path = r"C:\Users\edoua\Desktop\Essai 2_output - frame at 20m54s.jpg"
image = cv2.imread(image_path)

model = YOLO(r"C:\Users\edoua\Documents\Birse\Bristol\MSc Thesis\Modèle de Machine Learning\1-YOLO-V8-Classifier\runs\content\runs\classify\train\weights\best.pt")

result = model(image)[0].probs

cv2.putText(image, 'INSIDE PLUME', (110, 320), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

if float(result.data[0]) > 0.5:
    print("We are inside the plume")
else:
    print("We are outside the plume")


cv2.imshow('img',image)







