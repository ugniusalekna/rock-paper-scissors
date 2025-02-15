import cv2 as cv
import numpy as np
import onnxruntime as ort


class RPSInference:
    def __init__(self, model_path, class_map):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.image_size = self.input_shape[-2:]
        self.reverse_class_map = {v: k for k, v in class_map.items()}

    def preprocess(self, frame):
        frame = cv.resize(frame, self.image_size)
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        frame = np.expand_dims(frame, axis=0)
        return frame
    
    def forward(self, frame):
        inputs = {self.input_name: frame}
        outputs = self.session.run([self.output_name], inputs)
        return np.argmax(outputs[0], axis=1)[0]

    def predict(self, frame):
        preprocessed_frame = self.preprocess(frame)
        class_idx = self.forward(preprocessed_frame)
        return self.reverse_class_map[class_idx]