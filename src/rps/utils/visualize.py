import matplotlib.pyplot as plt


def show_image_grid(dataset, class_map, num_images=16, title=''):
    reverse_class_map = {v: k for k, v in class_map.items()}

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(title, fontsize=16)
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            img, label = dataset[i]
            ax.imshow(img.permute(1, 2, 0))
            ax.set_title(f'Label: {reverse_class_map[label]}')
            ax.axis('off')
    plt.show()