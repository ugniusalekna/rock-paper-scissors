#!/usr/bin/env python

from types import SimpleNamespace
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from rps.augment import Augmenter
from rps.dataset import RPSDataset
from rps.model import RPSClassifier
from rps.engine import Trainer, Validator
from rps.utils.data import load_yaml, get_image_paths, make_class_map, train_val_split
from rps.utils.misc import get_device, initialize_run_dir


def main():
    cfg_dict = load_yaml(path="../configs/config.yaml")
    cfg = SimpleNamespace(**cfg_dict)

    img_paths, class_map = get_image_paths(cfg.data_dir), make_class_map(cfg.data_dir)
    train_img_paths, val_img_paths = train_val_split(img_paths, val_ratio=cfg.val_ratio, seed=cfg.seed)

    train_dataset = RPSDataset(
        img_paths=train_img_paths, 
        class_map=class_map, 
        transform=Augmenter(train=True, image_size=cfg.image_size), 
    )
    val_dataset = RPSDataset(
        img_paths=val_img_paths,
        class_map=class_map,
        transform=Augmenter(train=False, image_size=cfg.image_size), 
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    device = get_device()
    model = RPSClassifier(num_classes=cfg.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    run_dir = initialize_run_dir(cfg.log_dir)
    writer = SummaryWriter(log_dir=run_dir)
    trainer = Trainer(model, optimizer, criterion, device, writer)
    validator = Validator(model, criterion, device, writer)

    best_val_acc = 0.0
    pbar = tqdm(range(1, cfg.num_epochs + 1), leave=False)
    
    for epoch in pbar:
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
        val_loss, val_acc = validator.validate(val_loader, epoch)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{run_dir}/best_model.pt")
        
        pbar.set_postfix_str(f"Loss {train_loss:.4f}, Acc {train_acc:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")
    
    writer.close()


if __name__ == '__main__':
    main()