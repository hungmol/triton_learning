from yolov5.YOLOv5 import YOLOv5
from yolov8.YOLOv8 import YOLOv8
import cv2


def main(input_type, source, model_name, triton_url="localhost:8000"):
    if model_name == 'yolov8':
        yolo = YOLOv8()
    else: 
        yolo = YOLOv5()
        
    if input_type == "image":
        img = cv2.imread(source)
        img_with_bboxes = yolo.infer(img, model_name)
        cv2.imshow("Inference Result", img_with_bboxes)
        cv2.waitKey(0)

    elif input_type == "video" or input_type == "camera":
        cap = cv2.VideoCapture(source if input_type == "video" else 0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            img_with_bboxes = yolo.infer(frame, model_name)
            cv2.imshow("Inference Result", img_with_bboxes)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # For image: main("image", "path_to_image.jpg")
    # For video: main("video", "path_to_video.mp4")
    # For camera: main("camera", "")
    main(input_type="video", source="test.mp4", model_name="yolov8")
