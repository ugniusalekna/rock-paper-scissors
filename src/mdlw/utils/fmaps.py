import math
import cv2 as cv
import numpy as np
import torchvision.transforms as T


def get_hook(activation, layer_name):
    def hook(module, inp, output):
        activation[layer_name] = output.detach()
    return hook


def build_grid(feature_tensor):
    n, h, w = feature_tensor.shape

    grid_cols = math.ceil(math.sqrt(n))
    grid_rows = math.ceil(n / grid_cols)
    grid = np.zeros((grid_rows * h, grid_cols * w), dtype=np.uint8)
    
    for idx in range(n):
        row, col = idx // grid_cols, idx % grid_cols
        fmap = feature_tensor[idx]

        fmap_np = fmap.cpu().numpy()
        fmap_np = (fmap_np - fmap_np.min()) / (fmap_np.ptp() + 1e-5) * 255
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = fmap_np.astype(np.uint8)
    
    return grid


def build_fc_composite(activation, fc_layers, target_width, target_height):
    stripes = []
    stripe_height = target_height // len(fc_layers) if len(fc_layers) > 0 else target_height
    
    for name in fc_layers:
        if name not in activation:
            stripe = np.zeros((1, target_width), dtype=np.uint8)
        else:
            fc_out = activation[name].squeeze(0)
            fc_np = fc_out.cpu().numpy()
            fc_np = (fc_np - fc_np.min()) / (fc_np.ptp() + 1e-5) * 255
            fc_np = fc_np.astype(np.uint8)
            stripe = cv.resize(fc_np[np.newaxis, :], (target_width, stripe_height), interpolation=cv.INTER_NEAREST)
        stripe_colored = cv.applyColorMap(stripe, cv.COLORMAP_VIRIDIS)
        stripes.append(stripe_colored)

    composite = np.vstack(stripes)
    return composite


def preprocess(image, imgsz, device='cpu'):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((imgsz, imgsz)),
    ])
    
    return transform(image).to(device)


def update_hooks(model, activation, layer_names):
    found_layers = set()
    
    for module in model.modules():
        if hasattr(module, 'hook_handle'):
            module.hook_handle.remove()
            del module.hook_handle
    
    for name, module in model.named_modules():
        if name in layer_names:
            module.hook_handle = module.register_forward_hook(get_hook(activation, name))
            found_layers.add(name)
            if found_layers == set(layer_names):
                return