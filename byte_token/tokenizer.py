from numpy import array
from PIL import Image

from byte_token.process_data import load_data
from config import Config

cfg = Config()


class ByteLevelTokenizer:
    def __init__(self):
        trainset, testset, classes = load_data()
        self.trainset = trainset
        self.testset = testset
        self.classes = classes

    def save_img(self, img):
        img = Image.fromarray(array(img.cpu()), mode="P")
        img.putpalette(self.palette_img.getpalette())
        img.save("image.png")

    def get_train_dataloader(self):
        return self.trainset

    def get_test_dataloader(self):
        return self.testset
