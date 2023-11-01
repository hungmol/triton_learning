import time
import cv2
import numpy as np
import tritonclient.http as tritonhttpclient
from yolov8.utils import xywh2xyxy, draw_detections, multiclass_nms


class YOLOv8:

    def __init__(self, triton_url="localhost:8000", conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.input_height = 480
        self.input_width = 640
        self.input_shape = [1, 3, 480, 640]
        
        self.triton_url = triton_url
        self.client = tritonhttpclient.InferenceServerClient(url=self.triton_url, verbose=False)


    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def infer (self,  img, model_name):
        # Preprocess
        preprocessed = self.prepare_input(img)
        inputs = [tritonhttpclient.InferInput('images', self.input_shape, 'FP32')]
        inputs[0].set_data_from_numpy(preprocessed)

        # Inference
        response = self.client.infer(model_name, inputs)
        output_data = response.as_numpy("output0")
        # Extract bounding boxes, classes, and confidences from output_data
        # Note: This depends on your model's output format
        # Decode the output tensor data
        self.boxes, self.scores, self.class_ids = self.process_output(output_data)
        combined_img = self.draw_detections(img)
        return combined_img


    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):

        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


# if __name__ == '__main__':
#     from imread_from_url import imread_from_url

#     model_path = "../models/yolov8m.onnx"

#     # Initialize YOLOv8 object detector
#     yolov8_detector = YOLOv8(model_path, conf_thres=0.3, iou_thres=0.5)

#     img_url = "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg"
#     img = imread_from_url(img_url)

#     # Detect Objects
#     yolov8_detector(img)

#     # Draw detections
#     combined_img = yolov8_detector.draw_detections(img)
#     cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
#     cv2.imshow("Output", combined_img)
#     cv2.waitKey(0)
