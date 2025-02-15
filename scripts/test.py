#!/usr/bin/env python

import argparse
import cv2 as cv

from rps.inference import RPSInference
from rps.utils.data import make_class_map
from rps.utils.capture import video_capture, crop_square, draw_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='../data')
    return parser.parse_args()


def main():
    args = parse_args()
    class_map = make_class_map(args.data_dir)
    model = RPSInference(model_path=args.onnx_path, class_map=class_map)

    with video_capture() as cap:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            frame = crop_square(frame)
            display_frame = cv.flip(frame.copy(), 1)
            
            prediction = model.predict(frame)
            draw_text(display_frame, text=f"Prediction: {prediction}", font_scale=1.0, pos=(10, 50))

            cv.imshow('Inference', display_frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main()
