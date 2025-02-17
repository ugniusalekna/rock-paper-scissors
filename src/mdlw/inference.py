import cv2 as cv
import numpy as np
import onnxruntime as ort


class InferenceModel:
    def __init__(self, model_path, class_map):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.image_size = self.input_shape[-2:]
        self.reverse_class_map = {v: k for k, v in class_map.items()}

    def preprocess(self, frame): # Expects 1:1 aspect ratio
        frame = cv.resize(frame, self.image_size)
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        frame = np.expand_dims(frame, axis=0)
        return frame
    
    def predict(self, frame, return_prob=False):
        frame = self.preprocess(frame)
        outputs = self.session.run([self.output_name], {self.input_name: frame})[0]
        class_idx = np.argmax(outputs)
        if return_prob:
            probs = np.exp(outputs - np.max(outputs)) / np.sum(np.exp(outputs - np.max(outputs)))
            return self.reverse_class_map[class_idx], probs[class_idx]
        return self.reverse_class_map[class_idx]