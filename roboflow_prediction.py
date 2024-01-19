import cv2
from ultralytics import YOLO
import supervision as sv

# Open the video file
video_path = r"C:\Users\edoua\Documents\Birse\Bristol\MSc Thesis\ExampleofFlight\2019-03-25_5_LukeFuckingAmazingLoadsofAshFalling\GoPro Front\full_flight_640.mp4"


def main():
    # to save the video
    writer= cv2.VideoWriter(r'C:\Users\edoua\Documents\Birse\Bristol\MSc Thesis\ExampleofFlight\2019-03-25_5_LukeFuckingAmazingLoadsofAshFalling\GoPro Front\output_yolo.mp4',
                            cv2.VideoWriter_fourcc(*'DIVX'),
                            7,
                            (640, 640))

    # define resolution
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    # specify the model
    model = YOLO(r'C:\Users\edoua\Documents\Birse\Bristol\MSc Thesis\Modèle de Machine Learning\YOLO-V8-200-995 frames\best.pt')

    # customize the bounding box
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )


    while True:
        ret, frame = cap.read()
        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
            f"{model.model.names[class_id]} {confidence:0.5f}"
            for _, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        writer.write(frame)

        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27): # break with escape key
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()