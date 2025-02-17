import torchvision.transforms as T
import torchvision.transforms.v2 as T2


class Augmenter:
    def __init__(self, train=True, image_size=224):
        if train:
            self.transforms = T.Compose([
                T.ToTensor(),
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomPerspective(distortion_scale=0.2, p=0.5),
                # T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                # T.RandomGrayscale(p=0.2),
                # T.GaussianBlur(kernel_size=(3, 3), sigma=(0.01, 1.0)),
                T.RandomApply([T2.GaussianNoise(sigma=0.01)], p=1.0),
            ])
        else:
            self.transforms = T.Compose([
                T.ToTensor(),
                T.Resize((image_size, image_size)),
            ])

    def __call__(self, img):
        return self.transforms(img)