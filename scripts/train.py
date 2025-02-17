from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mdlw.augment import Augmenter
from mdlw.dataset import ImageDataset
from mdlw.model import ImageClassifier as ImageClassifier
from mdlw.engine import Trainer, Validator, Exporter
from mdlw.utils.data import get_image_paths, make_class_map, train_val_split
from mdlw.utils.misc import load_cfg, get_device, initialize_run_dir, save_cfg


def main():
    cfg = load_cfg(path="../configs/config.yaml")
    run_dir = initialize_run_dir(cfg.log_dir)
    save_cfg(cfg, f"{run_dir}/args.yaml")
    
    img_paths, class_map = get_image_paths(cfg.data_dir), make_class_map(cfg.data_dir)
    train_img_paths, val_img_paths = train_val_split(img_paths, val_ratio=cfg.val_ratio, seed=cfg.seed)

    train_dataset = ImageDataset(
        img_paths=train_img_paths, 
        class_map=class_map, 
        transform=Augmenter(train=True, image_size=cfg.image_size), 
    )
    val_dataset = ImageDataset(
        img_paths=val_img_paths,
        class_map=class_map,
        transform=Augmenter(train=False, image_size=cfg.image_size), 
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    device = get_device()
    model = ImageClassifier(num_classes=cfg.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.learning_rate, steps_per_epoch=len(train_loader), epochs=cfg.num_epochs)
    scheduler = None
    loss_fn = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=run_dir)
    trainer = Trainer(model, optimizer, scheduler=scheduler, criterion=loss_fn, device=device, writer=writer)
    validator = Validator(model, criterion=loss_fn, device=device, writer=writer)

    best_val_acc = 0.0
    pbar = tqdm(range(1, cfg.num_epochs + 1), leave=False)
    
    for epoch in pbar:
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
        val_loss, val_acc = validator.validate(val_loader, epoch)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, f"{run_dir}/best_model.pt")
        
        pbar.set_postfix_str(f"Loss {train_loss:.4f}, Acc {train_acc:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")
    
    writer.close()
    exporter = Exporter(model, imgsz=cfg.image_size, device=device)
    exporter.export_onnx(f"{run_dir}/best_model.onnx")


if __name__ == '__main__':
    main()