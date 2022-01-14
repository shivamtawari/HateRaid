import json
import os
import random
import numpy as np

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

import clip
import pickle

import easyocr

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    MMBTConfig,
    MMBTModel,
    MMBTForClassification,
    get_linear_schedule_with_warmup,
)


class ClipEncoderMulti(nn.Module):
    def __init__(self, clip_model, num_embeds, num_features):
        super().__init__()        
        self.model = clip_model
        self.num_embeds = num_embeds
        self.num_features = num_features

    def forward(self, x):
        # 4x3x288x288 -> 1x4x640
        out = self.model.encode_image(x.view(-1,3,288,288))
        out = out.view(-1, self.num_embeds, self.num_features).float()
        return out  # Bx4x640


class Prepucess:
    def __init__(self, tokenizer, transforms, max_seq_length, image_encoder_size=288, device="cpu"):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.transforms = transforms

        self.image_encoder_size = image_encoder_size
        self.device = device

    def process(self, img_dir, text):
        sentence = torch.LongTensor(self.tokenizer.encode(text, add_special_tokens=True))
        start_token, sentence, end_token = sentence[0], sentence[1:-1], sentence[-1]
        sentence = sentence[:self.max_seq_length]

        image = Image.open(img_dir).convert("RGB")
        sliced_images = slice_image(image, 288)
        sliced_images = [np.array(self.transforms(im)) for im in sliced_images]
        image = resize_pad_image(image, self.image_encoder_size)
        image = np.array(self.transforms(image))
        
        sliced_images = [image] + sliced_images         
        sliced_images = torch.from_numpy(np.array(sliced_images)).to(self.device)

        return {
            "image_start_token": start_token,            
            "image_end_token": end_token,
            "sentence": sentence,
            "image": sliced_images          
        }

def final_collate_fn(batch):
    # lens = [len(row["sentence"]) for row in batch]
    lens = [len(batch["sentence"])]
    bsz, max_seq_len = 1, max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
    text_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)

    text_tensor[0, :lens[0]] = batch["sentence"]
    mask_tensor[0, :lens[0]] = 1

    # img_tensor = torch.stack([row["image"] for row in batch])
    img_tensor = torch.tensor(batch["image"])
    id_tensor = torch.tensor([76432])
    img_start_token = torch.tensor(batch["image_start_token"])
    img_end_token = torch.tensor(batch["image_end_token"])

    return (
        text_tensor, mask_tensor, img_tensor[None], 
        img_start_token[None], img_end_token[None], id_tensor[None]
    )

def slice_image(im, desired_size):
    '''
    Resize and slice image
    '''
    old_size = im.size

    ratio = float(desired_size) / min(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)

    ar = np.array(im)
    images = []
    if ar.shape[0] < ar.shape[1]:
        middle = ar.shape[1] // 2
        half = desired_size // 2

        images.append(Image.fromarray(ar[:, :desired_size]))
        images.append(Image.fromarray(ar[:, middle - half:middle + half]))
        images.append(Image.fromarray(ar[:, ar.shape[1] - desired_size:ar.shape[1]]))
    else:
        middle = ar.shape[0] // 2
        half = desired_size // 2

        images.append(Image.fromarray(ar[:desired_size, :]))
        images.append(Image.fromarray(ar[middle - half:middle + half, :]))
        images.append(Image.fromarray(ar[ar.shape[0] - desired_size:ar.shape[0], :]))

    return images

def resize_pad_image(im, desired_size):
    '''
    Resize and pad image to a desired size
    '''
    old_size = im.size

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)

    # create a new image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))

    return new_im


class OCR:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def recognize(self, img_path):
        txt = self.reader.readtext(img_path, detail = 0)
        clean = ' '.join(txt)
        return clean


class Model:
    def __init__(self, model_path='models/model.pt'):
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        self.num_image_embeds = 4
        self.num_labels = 1
        self.max_seq_length = 80 
        self.max_grad_norm = 0.5
        self.image_encoder_size = 288
        self.image_features_size = 640

        self.clip_model, self.preprocess = clip.load("RN50x4", device=self.device, jit=False)

        for p in self.clip_model.parameters():
            p.requires_grad = False

        model_name = 'Hate-speech-CNERG/bert-base-uncased-hatexplain'
        transformer_config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=transformer_config)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
        self.image_encoder = ClipEncoderMulti(self.clip_model, self.num_image_embeds, self.image_features_size)

        self.pre = Prepucess(self.tokenizer, self.preprocess, self.max_seq_length - self.num_image_embeds - 2)

        config = MMBTConfig(transformer_config, num_labels=self.num_labels,
                            modal_hidden_size=self.image_features_size)
        self.model = MMBTForClassification(config, self.transformer, self.image_encoder)
        _ = self.load_checkpoint(model_path, self.model)

        self.ocr = OCR()


    def load_checkpoint(self, load_path, model):
        if load_path==None:
            return
    
        state_dict = torch.load(load_path, map_location=self.device)
        print(f'Model loaded from <== {load_path}')
    
        model.load_state_dict(state_dict['model_state_dict'])
        return state_dict['valid_loss']

    def final_prediction(self, batch, thres=0.5):
        preds = None
        proba = None

        self.model.eval()
        batch = tuple(t.to(self.device) for t in batch)

        with torch.no_grad():
            ids = batch[5]
            inputs = {
                "input_ids": batch[0],
                "input_modal": batch[2],
                "attention_mask": batch[1],
                "modal_start_tokens": batch[3],
                "modal_end_tokens": batch[4],
                "return_dict": False
            }
            outputs = self.model(**inputs)
            logits = outputs[0]

            preds = torch.sigmoid(logits).detach().cpu().numpy() > thres
            proba = torch.sigmoid(logits).detach().cpu().numpy()
    
        result = {
            "preds": preds,
            "probs": proba,
        }
    
        return result

    def predict(self, image_dir=None, text=None):
        if (image_dir==None and text==None):
            print('Give this model something!')
            return -1

        if image_dir:
            onimage = self.ocr.recognize(image_dir)
            thres = 0.5
            if text:
                text = text + ' ' + onimage
                batch = self.pre.process(image_dir, text)
            batch = self.pre.process(image_dir, onimage)
        else:
            image_dir = os.path.join(os.getcwd(), 'static', 'img', 'memes', 'black.png')
            thres = 0.25

            batch = self.pre.process(image_dir, text)

        fin_bat = final_collate_fn(batch)

        results = self.final_prediction(fin_bat, thres)
        print(results)

        return results['preds'][0][0]

if __name__ == '__main__':
    model = Model()

    #print(model.predict('40916.png', 'my president sitting with a traitor and his tranny wife'))
    print(model.predict('40916.png'))
