from torch.utils.data import Dataset
from .utils.data import get_cls_from_path, read_image


class RPSDataset(Dataset):
    def __init__(self, img_paths, class_map, transform=None):
        self.img_paths = img_paths
        self.class_map = class_map
        self.transform = transform
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        cls_name = get_cls_from_path(img_path)
        label = self.class_map[cls_name]
        img = read_image(img_path)
        
        if self.transform:
            img = self.transform(img)

        return img, label