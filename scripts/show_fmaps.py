import argparse
import torch
import cv2 as cv
import numpy as np

from mdlw.model import ImageClassifier
from mdlw.utils.capture import video_capture, draw_text, crop_square
from mdlw.utils.fmaps import update_hooks, build_grid, preprocess, build_fc_composite
from mdlw.utils.misc import load_cfg, get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--config_path', type=str, default="../configs/config.yaml")
    p.add_argument('--layer', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_cfg(path=args.config_path)

    device = get_device()
    model = ImageClassifier(num_classes=cfg.num_classes).to(device)
    model.eval()
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    
    conv_layers = [name for name, _ in model.named_modules() if name.startswith("conv")]
    fc_layers = [name for name, _ in model.named_modules() if name.startswith("fc")]
    
    if not conv_layers:
        raise ValueError("No conv layers found")
    
    layers = conv_layers + ['fc_composite']
    
    current_idx = 0
    if args.layer and args.layer in conv_layers:
        current_idx = conv_layers.index(args.layer)
    current_layer = layers[current_idx]
    
    activation = {}
    if current_layer == 'fc_composite':
        update_hooks(model, activation, fc_layers)
    else:
        update_hooks(model, activation, [current_layer])
    
    def process(image):
        nonlocal activation, current_idx, layers
        image = cv.flip(crop_square(image), 1)
        tensor = preprocess(image, imgsz=cfg.image_size, device=device).unsqueeze(0)
        _ = model(tensor)
        
        if layers[current_idx] == 'fc_composite':
            vis = build_fc_composite(activation, fc_layers, image.shape[1], image.shape[0])
            draw_text(vis, text="FC layers", pos=(10, 30), font_scale=1.0)
        else:
            feat = activation[layers[current_idx]].squeeze(0)
            grid = build_grid(feat)
            grid_resized = cv.resize(grid, (image.shape[1], image.shape[0]), interpolation=cv.INTER_NEAREST)
            vis = cv.applyColorMap(grid_resized, cv.COLORMAP_VIRIDIS)
            draw_text(vis, text=layers[current_idx], pos=(10, 30), font_scale=1.0)
        
        return np.concatenate([image, vis], axis=1)
    
    paused = False
    with video_capture(0) as cap:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
            display = process(frame)
            draw_text(display, text="Press 'n' for next layer, 'p' for previous, 'q' to quit.", font_scale=1.0)
            cv.imshow("Feature visualization", display)
            
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                current_idx = (current_idx + 1) % len(layers)
                current_layer = layers[current_idx]
                if current_layer == 'fc_composite':
                    update_hooks(model, activation, fc_layers)
                else:
                    update_hooks(model, activation, [current_layer])
            elif key == ord('p'):
                current_idx = (current_idx - 1) % len(layers)
                current_layer = layers[current_idx]
                if current_layer == 'fc_composite':
                    update_hooks(model, activation, fc_layers)
                else:
                    update_hooks(model, activation, [current_layer])
            elif key == ord(' '):
                paused = not paused
                print("Paused" if paused else "Resumed")


if __name__ == '__main__':
    main()