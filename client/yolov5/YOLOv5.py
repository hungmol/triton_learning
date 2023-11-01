import cv2
import numpy as np
import tritonclient.http as tritonhttpclient


class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", 
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", 
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))

class YOLOv5:
    def __init__(self, triton_url="localhost:8000", conf_threshold=0.5):
        self.triton_url = triton_url
        self.client = tritonhttpclient.InferenceServerClient(url=self.triton_url, verbose=False)
        self.input_shape = [1, 3, 640, 640]
        self.conf_threshold = conf_threshold
    
    def preprocess_image(self, img):
        # Convert image from BGR to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize the image to the expected input shape (640, 640)
        img = cv2.resize(img, (self.input_shape[3], self.input_shape[2]))
        
        # Normalize the image (divide by 255 to bring values between 0 and 1)
        img = img / 255.0
        
    # Convert image to float32 and add batch dimension
        img = np.expand_dims(img, axis=0).astype(np.float32)
        
        # Transpose image dimensions to [batch, channels, height, width]
        img = np.transpose(img, (0, 3, 1, 2))
        
        return img

    def iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        xx1, yy1, xx2, yy2 = box2

        inter_width = min(x2, xx2) - max(x1, xx1)
        inter_height = min(y2, yy2) - max(y1, yy1)

        if inter_width < 0 or inter_height < 0:
            intersection = 0
        else:
            intersection = inter_width * inter_height

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (xx2 - xx1) * (yy2 - yy1)

        union = area1 + area2 - intersection

        return intersection / union

    def non_max_suppression(self, boxes, scores, threshold):
        # # Extract coordinates from boxes
        # x1 = [box[0] for box in boxes]
        # y1 = [box[1] for box in boxes]
        # x2 = [box[2] for box in boxes]
        # y2 = [box[3] for box in boxes]
        
        # Initialize lists for suppressed boxes and indices
        suppressed = set()
        keep = []
        
        # Sort scores and corresponding boxes in descending order
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        for idx in order:
            if idx in suppressed:
                continue
            
            # This box is not suppressed, so we'll keep it
            keep.append(idx)
            
            # Suppress other boxes with IOU greater than the threshold
            for other_idx in order:
                if other_idx == idx or other_idx in suppressed:
                    continue
                overlap = self.iou(boxes[idx], boxes[other_idx])
                if overlap > threshold:
                    suppressed.add(other_idx)
                    
        return keep

    def draw_boxes(self, image, boxes, confidences, class_labels, class_names, nms_threshold=0.3):
        # Convert boxes to (x1, y1, x2, y2) format
        boxes_xy = []
        
        h, w = image.shape[:2]
        xGain = self.input_shape[3] / w
        yGain = self.input_shape[2] / h

        for box in boxes:
            x_center, y_center, width, height = box
            x1 = int((x_center - width / 2) / xGain)
            y1 = int((y_center - height / 2) / yGain)
            x2 = int((x_center + width / 2) / xGain)    
            y2 = int((y_center + height / 2) / yGain)
            
            boxes_xy.append((x1, y1, x2, y2))
        
        indices = self.non_max_suppression(boxes_xy, confidences, nms_threshold)

        mask_img = image.copy()
        mask_alpha = 0.3
        for index in indices:
            x1, y1, x2, y2 = boxes_xy[index]
            conf = confidences[index]
            label = class_labels[index]
            color = colors[label]
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), [255,255,0], 2)
            cv2.putText(image, f'{class_names[label.item()]} {conf.item():.2f}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)

    def decode_output(self, output_data):
        # Assuming output is of shape [1, 25200, 85]
        # 85 = 4 box coordinates + 1 objectness score + 80 class scores
        boxes = output_data[0, :, :4]
        confidences = output_data[0, :, 4:5]
        class_probs = output_data[0, :, 5:]
        
        # Calculate class scores & class labels
        class_scores = np.max(class_probs, axis=1)
        class_labels = np.argmax(class_probs, axis=1)
        
        # Filter out detections with low confidence
        mask = confidences[:, 0] > self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_scores = class_scores[mask]
        class_labels = class_labels[mask]
        
        return boxes, confidences, class_scores, class_labels


    def infer(self, img, model_name):
    
        # Preprocess
        preprocessed = self.preprocess_image(img)
        inputs = [tritonhttpclient.InferInput('images', self.input_shape, 'FP32')]
        inputs[0].set_data_from_numpy(preprocessed)

        # Inference
        response = self.client.infer(model_name, inputs)
        output_data = response.as_numpy("output0")
        # Extract bounding boxes, classes, and confidences from output_data
        # Note: This depends on your model's output format
        # Decode the output tensor data
        boxes, confidences, class_scores, class_labels = self.decode_output(output_data)
        
        # Draw bounding boxes
        # img_with_bboxes = draw_bboxes(img, bboxes, classes, confidences)
        image_with_boxes = self.draw_boxes(img, boxes, confidences, class_labels, class_names)
        return image_with_boxes