from PIL import Image
from torch.utils.data import Dataset


class Caltech101(Dataset):
    def __init__(self, img_path: list, img_class: list, transform=None):
        self.img_path = img_path
        self.img_class = img_class
        self.transform = transform

    def __getitem__(self, item):
        img = Image.open(self.img_path[item]).convert("RGB")
        label = self.img_class[item]
        if img.mode != "RGB":
            raise ValueError("Image: {} isn't RGB mode!".format(self.img_path[item]))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img_path)
