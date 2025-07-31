import cv2
import numpy as np
from tensorflow.keras.models import load_model

class Captcha(object):
    def __init__(self):
        self.model = load_model("models/captcha_character_model.h5")
        self.char_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        self.char_count = 5

    def preprocess_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray.astype("float32") / 255.0
        return gray

    def segment_characters(self, img):
        h, w = img.shape
        char_width = w // self.char_count
        characters = []
        for i in range(self.char_count):
            ch = img[:, i*char_width:(i+1)*char_width]
            ch = cv2.resize(ch, (28, 28))
            ch = np.expand_dims(ch, axis=(0, -1))
            characters.append(ch)
        return characters

    def predict_character(self, ch_img):
        pred = self.model.predict(ch_img)
        return self.char_list[np.argmax(pred)]

    def __call__(self, im_path, save_path):
        img = cv2.imread(im_path)
        proc = self.preprocess_image(img)
        segments = self.segment_characters(proc)
        result = ''.join([self.predict_character(ch) for ch in segments])
        with open(save_path, 'w') as f:
            f.write(result + '\n')