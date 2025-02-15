#!/usr/bin/env python

import argparse
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader

from rps.augment import Augmenter
from rps.dataset import RPSDataset
from rps.model import RPSClassifier
from rps.engine import Validator
from rps.utils.data import load_yaml, get_image_paths, make_class_map, train_val_split
from rps.utils.misc import get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, required=True)
    return p.parse_args()


def main():
    cli_args = parse_args()
    cfg_dict = load_yaml(path="../configs/config.yaml")
    cfg = SimpleNamespace(**cfg_dict)
    
    img_paths, class_map = get_image_paths(cfg.data_dir), make_class_map(cfg.data_dir)
    _, val_img_paths = train_val_split(img_paths, val_ratio=cfg.val_ratio, seed=cfg.seed)

    val_dataset = RPSDataset(
        img_paths=val_img_paths,
        class_map=class_map,
        transform=Augmenter(train=False, image_size=cfg.image_size), 
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    device = get_device()
    model = RPSClassifier(num_classes=cfg.num_classes).to(device)
    model.load_state_dict(torch.load(cli_args.model_path, map_location=device, weights_only=True))

    criterion = torch.nn.CrossEntropyLoss()
    validator = Validator(model, criterion, device)
    
    print("Validating before exporting...")
    avg_loss, avg_acc = validator.validate(val_loader, epoch=0)
    print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {avg_acc:.4f}")
    
    print("Exporting model...")
    model.eval()
    dummy_input = torch.randn(1, 3, cfg.image_size, cfg.image_size, device=device)
    onnx_path = cli_args.model_path.replace('.pt', '.onnx')
    torch.onnx.export(
        model=model, 
        args=dummy_input, 
        f=onnx_path,
        input_names=["input"], 
        output_names=["output"], 
        opset_version=11
    )
    print(f"Model exported to {onnx_path}")


if __name__ == '__main__':
    main()