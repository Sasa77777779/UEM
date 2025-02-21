import json
import torch
import torch.utils.data as data
import numpy as np
import re
import h5py
import os
import pickle
import torch.nn.functional as F


def getVideoId(cap_id):
    vid_id = cap_id.split('#')[0]
    return vid_id

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip().lower().split()

def read_video_ids(cap_file):
    video_ids_list = []
    with open(cap_file, 'r') as cap_reader:
        for line in cap_reader.readlines():
            cap_id, caption = line.strip().split(' ', 1)
            video_id = getVideoId(cap_id)
            if video_id not in video_ids_list:
                video_ids_list.append(video_id)
    return video_ids_list

def compute_similarity(a, b):
    a_normalized = F.normalize(a, p=2, dim=-1)
    b_normalized = F.normalize(b, p=2, dim=-1)
    cosine_similarity = torch.dot(a_normalized, b_normalized)
    return cosine_similarity

def progressive_segmentation(X, threshold=0.9, max_frame_num=128):
    event_mask = torch.zeros(max_frame_num, max_frame_num)
    cluster_center = X[0]
    event_id = 0
    for i in range(X.shape[0]):
        cosine_similarity = compute_similarity(X[i], cluster_center)
        if cosine_similarity > threshold:
            event_mask[event_id][i] = 1.0
            cluster_center = (cluster_center + X[i]) / 2
        else:
            event_id = event_id + 1
            event_mask[event_id][i] = 1.0
            cluster_center = X[i]

    return event_mask

def uniform_feature_sampling(features, max_len):
    num_clips = features.shape[0]
    if max_len is None or num_clips <= max_len:
        return features
    idxs = np.arange(0, max_len + 1, 1.0) / max_len * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1
    new_features = []
    for i in range(max_len):
        s_idx, e_idx = idxs[i], idxs[i + 1]
        if s_idx < e_idx:
            new_features.append(np.mean(features[s_idx:e_idx], axis=0))
        else:
            new_features.append(features[s_idx])
    new_features = np.asarray(new_features)
    return new_features


def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)


def collate_train(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)

    video_event_mask, video_frame_features, captions, word_tokens, idxs, cap_ids, video_ids = zip(*data)

    video_event_masks = torch.cat(video_event_mask, dim=0).float()

    video_lengths = [len(frame) for frame in video_frame_features]
    frame_vec_len = len(video_frame_features[0][0])
    frame_videos = torch.zeros(len(video_frame_features), max(video_lengths), frame_vec_len)
    videos_mask = torch.zeros(len(video_frame_features), max(video_lengths))
    for i, frames in enumerate(video_frame_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0

    feat_dim = captions[0][0].shape[-1]
    merge_captions = []
    clip_labels = []

    for index, caps in enumerate(captions):
        clip_labels.extend(index for _ in range(len(caps)))
        merge_captions.extend(cap for cap in caps)

    clip_target = torch.zeros(len(merge_captions), feat_dim)

    for index, cap in enumerate(merge_captions):
        clip_target[index, :] = cap

    word_merge_captions = []
    word_all_lengths = []

    for index, tokens in enumerate(word_tokens):
        word_all_lengths.extend(len(token) for token in tokens)
        word_merge_captions.extend(token for token in tokens)

    words_target = torch.zeros(len(word_all_lengths), max(word_all_lengths), feat_dim)
    words_mask = torch.zeros(len(word_all_lengths), max(word_all_lengths))

    for index, tokens in enumerate(word_merge_captions):
        end = word_all_lengths[index]
        words_target[index, :end, :] = tokens[:end, :]
        words_mask[index, :end] = 1.0


    return dict(video_event=video_event_masks,
                video_frame_features=frame_videos,
                videos_mask=videos_mask,
                text_feat=clip_target,
                text_labels=clip_labels,
                word_feat=words_target,
                word_mask=words_mask
                )


def collate_frame_val(data):

    video_event_mask, video_frame_features, idxs, video_ids = zip(*data)

    video_event_masks = torch.cat(video_event_mask, dim=0).float()

    video_lengths = [len(frame) for frame in video_frame_features]
    frame_vec_len = len(video_frame_features[0][0])
    frame_videos = torch.zeros(len(video_frame_features), max(video_lengths), frame_vec_len)
    videos_mask = torch.zeros(len(video_frame_features), max(video_lengths))
    for i, frames in enumerate(video_frame_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0

    return video_event_masks, frame_videos, videos_mask, idxs, video_ids


def collate_text_val(data):
    if data[0][0] is not None:
        data.sort(key=lambda x: len(x[0]), reverse=True)
    captions, word_tokens, idxs, cap_ids = zip(*data)
    feat_dim = captions[0].shape[-1]
    clip_target = torch.zeros(len(captions), feat_dim)

    for index, caps in enumerate(captions):
        clip_target[index, :] = caps

    if word_tokens[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(token) for token in word_tokens]
        words_target = torch.zeros(len(word_tokens), max(lengths), word_tokens[0].shape[-1])
        words_mask = torch.zeros(len(word_tokens), max(lengths))
        for i, token in enumerate(word_tokens):
            end = lengths[i]
            words_target[i, :end] = token[:end]
            words_mask[i, :end] = 1.0
    else:
        words_target = None
        lengths = None
        words_mask = None

    return clip_target, words_target, words_mask, idxs, cap_ids


class Dataset4PRVR(data.Dataset):
    """
    Load captions and video frame features by pre-trained CNN model.
    """
    def __init__(self, cap_file, clip_vid_feat_path, clip_text_feat_path, clip_word_tokens_path, cfg, video2frames=None):
        # Captions
        self.captions = {}
        self.cap_ids = []
        self.video_ids = []
        self.vid_caps = {}
        self.video2frames = video2frames

        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                video_id = getVideoId(cap_id)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
                if video_id not in self.video_ids:
                    self.video_ids.append(video_id)
                if video_id in self.vid_caps:
                    self.vid_caps[video_id].append(cap_id)
                else:
                    self.vid_caps[video_id] = []
                    self.vid_caps[video_id].append(cap_id)

        self.clip_text_feat_path = clip_text_feat_path
        self.clip_word_tokens_path = clip_word_tokens_path
        self.clip_vid_feat_path = clip_vid_feat_path
        self.clip_text_feat = h5py.File(self.clip_text_feat_path, 'r')
        self.clip_word_tokens = h5py.File(self.clip_word_tokens_path, 'r')
        self.clip_vid_feat = h5py.File(self.clip_vid_feat_path, 'r')

        self.map_size = cfg['map_size']
        self.max_ctx_len = cfg['max_ctx_l']
        self.max_desc_len = cfg['max_desc_l']

        self.threshold = cfg['threshold']

        self.open_file = False
        self.length = len(self.vid_caps)


    def __getitem__(self, index):
        """
        if self.open_file:
            self.open_file = True
        else:
            self.text_feat = h5py.File(self.text_feat_path, 'r')

            self.open_file = True
        """
        video_id = self.video_ids[index]
        cap_ids = self.vid_caps[video_id]

        clip_vecs = []
        video_vecs = self.clip_vid_feat[video_id]
        for i in video_vecs:
            clip_vecs.append(i)

        video_frame_feature = uniform_feature_sampling(np.array(clip_vecs), self.max_ctx_len)
        video_frame_feature = torch.from_numpy(video_frame_feature)

        video_event_mask = progressive_segmentation(video_frame_feature, self.threshold, self.max_ctx_len)
        video_event_mask = video_event_mask.unsqueeze(0)

        clip_cap_tensors = []
        clip_word_tokens = []
        for cap_id in cap_ids:
            clip_cap_feat = self.clip_text_feat[cap_id][...]
            clip_cap_feat = torch.from_numpy(clip_cap_feat)
            clip_cap_tensors.append(clip_cap_feat)

            clip_word_feat = self.clip_word_tokens[cap_id][...]
            clip_word_tensor = torch.from_numpy(l2_normalize_np_array(clip_word_feat))[:self.max_desc_len]
            clip_word_tokens.append(clip_word_tensor)

        return video_event_mask, video_frame_feature, clip_cap_tensors, clip_word_tokens, index, cap_ids, video_id

    def __len__(self):
        return self.length


class VisDataSet4PRVR(data.Dataset):

    def __init__(self, clip_vid_feat_path, video2frames, cfg, video_ids=None):
        self.video2frames = video2frames
        if video_ids is not None:
            self.video_ids = video_ids
        else:
            self.video_ids = video2frames.keys()
        self.length = len(self.video_ids)
        self.map_size = cfg['map_size']
        self.max_ctx_len = cfg['max_ctx_l']
        self.threshold = cfg['threshold']
        self.clip_vid_feat_path = clip_vid_feat_path
        self.clip_vid_feat = h5py.File(self.clip_vid_feat_path, 'r')

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        clip_vecs = []
        video_vecs = self.clip_vid_feat[video_id]
        for i in video_vecs:
            clip_vecs.append(i)

        video_frame_feature = uniform_feature_sampling(np.array(clip_vecs), self.max_ctx_len)
        video_frame_feature = torch.from_numpy(video_frame_feature)

        video_event_mask = progressive_segmentation(video_frame_feature, self.threshold, self.max_ctx_len)
        video_event_mask = video_event_mask.unsqueeze(0)

        return video_event_mask, video_frame_feature, index, video_id

    def __len__(self):
        return self.length


class TxtDataSet4PRVR(data.Dataset):
    """
    Load captions
    """
    def __init__(self, cap_file, clip_text_feat_path, clip_word_tokens_path, cfg):
        # Captions
        self.captions = {}
        self.cap_ids = []
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
        self.clip_text_feat_path = clip_text_feat_path
        self.clip_word_tokens_path = clip_word_tokens_path
        self.max_desc_len = cfg['max_desc_l']
        self.open_file = False
        self.length = len(self.cap_ids)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        if self.open_file:
            self.open_file = True
        else:
            self.clip_text_feat = h5py.File(self.clip_text_feat_path, 'r')
            self.clip_word_tokens = h5py.File(self.clip_word_tokens_path, 'r')

            self.open_file = True

        clip_cap_feat = self.clip_text_feat[cap_id][...]
        clip_cap_feat = torch.from_numpy(clip_cap_feat)

        clip_word_tensors = self.clip_word_tokens[cap_id][...]
        clip_word_tokens = torch.from_numpy(l2_normalize_np_array(clip_word_tensors))[:self.max_desc_len]

        return clip_cap_feat, clip_word_tokens, index, cap_id

    def __len__(self):
        return self.length


if __name__ == '__main__':
    pass


