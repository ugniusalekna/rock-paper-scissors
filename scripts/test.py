import argparse
import cv2 as cv

from mdlw.inference import InferenceModel
from mdlw.utils.data import make_class_map
from mdlw.utils.capture import video_capture, crop_square, draw_text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--onnx_path', type=str, required=True)
    p.add_argument('--data_dir', type=str, default='../data/cifar10')
    return p.parse_args()


def main():
    args = parse_args()
    class_map = make_class_map(args.data_dir)
    model = InferenceModel(model_path=args.onnx_path, class_map=class_map)

    with video_capture() as cap:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            frame = crop_square(frame)
            display_frame = cv.flip(frame.copy(), 1)
            
            pred, prob = model.predict(frame, return_prob=True)
            draw_text(display_frame, text="Press 'q' to quit", font_scale=1.0, pos=(10, 40))
            draw_text(display_frame, text=f"Prediction: {pred}; Probability: {prob:.2f}", font_scale=1.0, pos=(10, 80))

            cv.imshow('Inference', display_frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main()