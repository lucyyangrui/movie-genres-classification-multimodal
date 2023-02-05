import json
import random
import time
import zipfile
from io import BytesIO
import os
from PIL import Image
import logging

import numpy as np
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch.nn.functional as F
from transformers import BertTokenizer, AutoFeatureExtractor
from transformers import Wav2Vec2Tokenizer
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
import jieba
import librosa
import pandas as pd
import math


def create_dataloader(args, train_index=None, val_index=None, test_mode=False):
    droplast = True
    if args.model_name == 'mymodel':
        droplast = False
        
    ann_path = args.annotation
    pic_video_path = args.pic_video_path

    logging.info('>>> loading data... from %s %s' % (ann_path, pic_video_path))
    dataset = MultiModalDataset(args, test_mode)
    if train_index is None and val_index is None:
        data_size = len(dataset)
        val_size = int(data_size * args.val_ratio)
        test_size = int(data_size * args.test_ratio)
        train_size = data_size - val_size - test_size
        index_shuffle = [i for i in range(data_size)]
        random.shuffle(index_shuffle)
        train_index = index_shuffle[: train_size]
        val_index = index_shuffle[train_size: train_size + val_size]
        test_index = index_shuffle[train_size + val_size: data_size]

    logging.info('>>> loading data... train_size: %d, val_size: %d, test_size %d' % (
        len(train_index), len(val_index), len(test_index)))
    train_dataset, val_dataset, test_dataset = torch.utils.data.Subset(dataset, train_index), \
                                               torch.utils.data.Subset(dataset, val_index), torch.utils.data.Subset(
        dataset, test_index),
    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = RandomSampler(val_dataset)
    test_sampler = RandomSampler(test_dataset)
    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.train_batch_size,
                                        sampler=train_sampler,
                                        drop_last=droplast)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=droplast)
    test_dataloader = dataloader_class(test_dataset,
                                       batch_size=args.val_batch_size,
                                       sampler=test_sampler,
                                       drop_last=droplast)

    train_index = pd.DataFrame(train_index)
    val_index = pd.DataFrame(val_index)
    test_index = pd.DataFrame(test_index)
    train_index.to_csv('./data/graph_data/train_index.csv', header=False, index=False)
    val_index.to_csv('./data/graph_data/val_index.csv', header=False, index=False)
    test_index.to_csv('./data/graph_data/test_index.csv', header=False, index=False)

    return train_dataloader, val_dataloader, test_dataloader


class MultiModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.

        Args:
            ann_path (str): annotation file path, with the '.json' suffix.
            pic_video_path (str): poster and video frames.
            test_mode (bool): if it's for testing.  -- no label
        """

    def __init__(self,
                 args,
                 # ann_path: str,
                 # pic_video_path: str,
                 test_mode: bool = False):
        self.model_name = args.model_name
        self.modals = args.modals
        self.max_frame = args.max_frame
        self.partial_num = args.partial_num
        self.partial_len = args.partial_len
        self.frequency_dimension = args.frequency_dimension
        self.temporal_dimension = args.temporal_dimension
        self.bert_seq_len = args.max_seq_len
        self.test_mode = test_mode

        self.cls_num = args.cls_num
        self.audio_feature_path = args.audio_feature_path
        
        self.audio_wav2vec_path = args.audio_wav2vec_path

        self.handles = [None for _ in range(args.num_workers)]

        self.pic_video_path = args.pic_video_path
        with open(args.annotation, 'r', encoding='utf-8') as f:
            self.anns = json.load(f)
            
        with open(self.audio_wav2vec_path, 'r', encoding='utf-8') as f:
            self.audio_feat_dict = json.load(f)
            
        with open(args.base1_audio_sus, 'r', encoding='utf-8') as f:
            self.base1_audio_sus = json.load(f)
            
        self.audio_part_num = args.audio_part_num

        # self.audio_tokenizer = Wav2Vec2Tokenizer.from_pretrained('facebook/wav2vec2-base-960h')

        def str_to_float(x):
            return float(x)

        self.word_vec_dict = {}
        with open(args.word2vec_path, 'rb') as f:
            for i, line_b in enumerate(f):
                line_u = line_b.decode('utf-8')
                if i >= 1:
                    word_vec = line_u.strip('\n ').split(' ')
                    self.word_vec_dict[word_vec[0]] = list(map(str_to_float, word_vec[1:]))

        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)

        self.visual_feature_exact = AutoFeatureExtractor.from_pretrained("microsoft/resnet-152",
                                                                         cache_dir=args.resnet_cache)
        self.transform = Compose([
            Resize(224),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.anns)

    def get_visual_frames(self, idx: int) -> tuple:
        vid = self.anns[idx]['movie_id']
        pic_video_path = os.path.join(self.pic_video_path, vid)
        movie_name = 'data' # os.listdir(pic_video_path)[0]
        video_frame_path = os.path.join(pic_video_path, movie_name, 'video_frames.zip')
        handler = zipfile.ZipFile(video_frame_path, 'r')
        namelist = sorted(handler.namelist(), key=lambda x: int(x.split('.')[0]))

        num_frames = len(namelist)

        frame = torch.zeros((self.max_frame, 3, 224, 224), dtype=torch.float32)
        mask = torch.zeros((self.max_frame,), dtype=torch.long)

        if num_frames <= self.max_frame:
            # load all frame
            select_inds = list(range(num_frames))
        else:
            select_inds = list(range(num_frames))
            random.shuffle(select_inds)
            select_inds = select_inds[:self.max_frame]
            select_inds = sorted(select_inds)

            # uniformly sample
            # step = num_frames // self.max_frame
            # select_inds = list(range(0, num_frames, step))
            # select_inds = select_inds[:self.max_frame]

            # randomly sample when test mode is False
            # select_inds = list(range(num_frames))
            # random.shuffle(select_inds)
            # select_inds = select_inds[:self.max_frame]
            # select_inds = sorted(select_inds)
        for i, j in enumerate(select_inds):
            mask[i] = 1
            img_content = handler.read(namelist[j])
            img = Image.open(BytesIO(img_content))
            if self.model_name == 'mymodel':
                img_tensor = self.transform(img)
            else:
                img_tensor = self.visual_feature_exact(img, return_tensors='pt')['pixel_values'][0]
            frame[i] = img_tensor

        return frame, mask

    def get_poster(self, idx: int):
        vid = self.anns[idx]['movie_id']
        pic_video_path = os.path.join(self.pic_video_path, vid)
        movie_name = os.listdir(pic_video_path)[0]
        poster_path = os.path.join(pic_video_path, movie_name, 'poster.webp')

        with Image.open(poster_path) as img:
            poster_tensor = self.transform(img.convert('RGB'))
        return poster_tensor

    def get_audio_feature(self, idx: int):
        vid = self.anns[idx]['movie_id']
        sus = self.base1_audio_sus[str(vid)]
        if sus:
            # zip_handler = zipfile.ZipFile(self.audio_feature_path, 'r')
            # BytesIO(zip_handler.read(name=str(vid) + '/audio_feature.npy'))
            audio_feature = np.load(self.audio_feature_path + '/' + str(vid) + '/audio_feature.npy')
        else:
            audio_mel_matric = []
            for i in range(5):
                mel_spect = torch.zeros((130, 513), dtype=torch.float)
                audio_mel_matric.append(np.array(mel_spect))
            audio_feature = np.array(audio_mel_matric)
        return torch.Tensor(audio_feature)

    def tokenizer_text(self, title: str, summary: str) -> tuple:
        title = title.split(' ')[0]
        summary = summary.replace(' ', '')
        text = '[SEP]' + title + '[SEP]' + summary
        encoded_input = self.tokenizer(text, max_length=self.bert_seq_len, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_input['input_ids'])
        mask = torch.LongTensor(encoded_input['attention_mask'])

        return input_ids, mask

    def get_text_vec(self, title: str, summary: str):
        title = title.split(' ')[0]
        summary = summary.replace(' ', '')
        text_words = jieba.cut(title + summary)
        word_vec = None
        len_text_word = 0
        for word in text_words:
            if word in self.word_vec_dict:
                len_text_word += 1
                if word_vec is None:
                    word_vec = torch.FloatTensor(self.word_vec_dict[word])
                else:
                    word_vec += torch.FloatTensor(self.word_vec_dict[word])
        if len_text_word == 0:
            word_vec = torch.zeros((300))
            len_text_word = 1
        word_vec /= len_text_word
        return word_vec

    def get_audio_wav2vec_feat(self, movie_id: str, audio_part_num=16):
        audio_wav2vec_feat = self.audio_feat_dict[movie_id]['audio_feature']
        mask = self.audio_feat_dict[movie_id]['mask']
        return torch.Tensor(audio_wav2vec_feat)[:], torch.LongTensor(mask)

    def __getitem__(self, idx: int) -> dict:
        data = {}
        if 'poster' in self.modals:
            poster_tensor = self.get_poster(idx)
            data['poster_input'] = poster_tensor
        if 'video' in self.modals:
            frame_tensor, frame_mask = self.get_visual_frames(idx)
            data['frame_input'] = frame_tensor
            data['frame_mask'] = frame_mask
        if 'summary' in self.modals:
            title = self.anns[idx]['title']
            summary = self.anns[idx]['summary']
            input_ids, input_mask = self.tokenizer_text(title, summary)
            data['text_input'] = input_ids
            data['text_mask'] = input_mask

        # word_vec = self.get_text_vec(title, summary)

        # audio_feature = self.get_audio_feature(idx)
        if 'audio' in self.modals:
            vid = self.anns[idx]['movie_id']
            audio_list, audio_mask = self.get_audio_wav2vec_feat(vid, self.audio_part_num)  # mymodel
            data['audio_wav_feat'] = audio_list
            data['audio_mask'] = audio_mask

        # data = dict(
        #     frame_input=frame_tensor,
        #     frame_mask=frame_mask,
        #     # poster_input=poster_tensor,
        #     text_input=input_ids,
        #     text_mask=input_mask,
        #     # word_vec=word_vec,
        #     # audio_feature=audio_feature,
        #     audio_wav_feat=audio_list,
        #     audio_mask=audio_mask
        # )

        if not self.test_mode:
            label = self.anns[idx]['types']
            label = torch.tensor(label)
            data['label'] = label
            data['label'] = torch.sum(F.one_hot(label, num_classes=self.cls_num), dim=0)

        return data

# test
# setup_logging()
# import config
# from utils import evaluate, setup_logging, setup_seed, setup_device, build_optimizer, build_optimizer_vlbert
# args = config.parse_args()
# setup_seed(args)
# setup_device(args)
# train_dataloader, val_dataloader, test_dataloader = create_dataloader(args)
# for i, t in enumerate(train_dataloader):
    # print(t['audio_list'].shape)
    # if i == 2:
        # break
