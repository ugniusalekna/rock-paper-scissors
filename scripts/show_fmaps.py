import argparse
import torch
import cv2 as cv
import numpy as np

from mdlw.utils.capture import video_capture, draw_text, crop_square
from mdlw.utils.data import make_class_map, reverse_class_map
from mdlw.utils.fmaps import update_hooks, build_grid, preprocess, build_fc_composite
from mdlw.utils.misc import load_cfg, get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--config_path', type=str, default="../configs/config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_cfg(path=args.config_path)
    
    class_map = make_class_map(cfg.data_dir)
    reversed_map = reverse_class_map(class_map)

    device = get_device()
    model = torch.load(args.model_path, map_location=device)
    model.eval()

    conv_layers = [name for name, _ in model.named_modules() if name.startswith("bn")]
    fc_layers = [name for name, _ in model.named_modules() if name.startswith("fc")]

    layers = conv_layers + ['fc_all']
    
    current_idx = 0
    current_layer = layers[current_idx]
    
    activation = {}
    update_hooks(model, activation, [current_layer])

    use_act = False

    def process(image, reversed_map):
        nonlocal activation, current_idx, layers, use_act
        image = cv.flip(crop_square(image), 1)
        tensor = preprocess(image, imgsz=cfg.image_size, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        class_idx = np.argmax(probs)
        pred, prob = reversed_map[class_idx], probs[class_idx]
        
        if layers[current_idx] == 'fc_all':
            vis = build_fc_composite(activation, fc_layers, gridsz=image.shape[:2], use_act=use_act)
            draw_text(vis, text="FC layers" + ('+act' if use_act else ''), pos=(10, 30), font_scale=1.0)
        else:
            feat = activation[layers[current_idx]].squeeze(0)
            grid = cv.resize(build_grid(feat, use_act=use_act), (image.shape[1], image.shape[0]), interpolation=cv.INTER_NEAREST)
            vis = cv.applyColorMap(grid, cv.COLORMAP_VIRIDIS)
            draw_text(vis, text=layers[current_idx] + ('+act' if use_act else ''), pos=(10, 30), font_scale=1.0)
        
        return np.concatenate([image, vis], axis=1), (pred, prob)
    
    paused = False
    with video_capture(0) as cap:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
            
            display, (pred, prob) = process(frame, reversed_map)
            draw_text(display, text="Press 'j', 'l' to change layers, 'k' to toggle act, 'q' to quit.", font_scale=1.0)
            draw_text(display, text=f"Prediction: {pred}: Probability: {prob:.2f}", font_scale=1.0, pos=(10, 80))
            cv.imshow("Feature visualization", display)
            
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                current_idx = (current_idx + 1) % len(layers)
                current_layer = layers[current_idx]
                hook_layers = fc_layers if current_layer == 'fc_all' else [current_layer]
                update_hooks(model, activation, hook_layers)
            elif key == ord('j'):
                current_idx = (current_idx - 1) % len(layers)
                current_layer = layers[current_idx]
                hook_layers = fc_layers if current_layer == 'fc_all' else [current_layer]
                update_hooks(model, activation, hook_layers)
            elif key == ord(' '):
                paused = not paused
            elif key == ord('k'):
                use_act = not use_act


if __name__ == '__main__':
    main()