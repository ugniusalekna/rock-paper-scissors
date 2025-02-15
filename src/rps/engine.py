import torch


class Trainer:
    def __init__(self, model, optimizer, criterion, device, writer=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.writer = writer

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        running_loss, running_corrects, total_samples = 0.0, 0, 0

        for i, (img_batch, label_batch) in enumerate(dataloader):
            img_batch, label_batch = img_batch.to(self.device), label_batch.to(self.device)
            self.optimizer.zero_grad()
            logit_batch = self.model(img_batch)
            loss = self.criterion(logit_batch, label_batch)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            pred_batch = logit_batch.argmax(dim=1)
            running_corrects += (pred_batch == label_batch).sum().item()
            total_samples += img_batch.size(0)

        avg_loss = running_loss / total_samples
        avg_acc = running_corrects / total_samples
        
        if self.writer:
            self.writer.add_scalar("Train/EpochLoss", avg_loss, epoch)
            self.writer.add_scalar("Train/EpochAcc", avg_acc, epoch)
            
        return avg_loss, avg_acc


class Validator:
    def __init__(self, model, criterion, device, writer=None):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.writer = writer

    def validate(self, dataloader, epoch):
        self.model.eval()
        running_loss, running_corrects, total_samples = 0.0, 0, 0

        with torch.no_grad():
            for img_batch, label_batch in dataloader:
                img_batch, label_batch = img_batch.to(self.device), label_batch.to(self.device)
                logit_batch = self.model(img_batch)
                loss = self.criterion(logit_batch, label_batch)
                running_loss += loss.item()
                pred_batch = logit_batch.argmax(dim=1)
                running_corrects += (pred_batch == label_batch).sum().item()
                total_samples += img_batch.size(0)

        avg_loss = running_loss / total_samples
        avg_acc = running_corrects / total_samples
        
        if self.writer:
            self.writer.add_scalar("Val/EpochLoss", avg_loss, epoch)
            self.writer.add_scalar("Val/EpochAcc", avg_acc, epoch)

        return avg_loss, avg_acc
