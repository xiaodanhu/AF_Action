import os
import json
import numpy as np
from random import shuffle
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from PIL import Image

from .datasets import register_dataset
from .data_utils import truncate_feats, get_transforms, truncate_video, circ_slice
from collections import defaultdict
import math

@register_dataset("finegym_uniform")
class FineGymDataset(Dataset):
    def __init__(
        self,
        is_training,     # if in training mode
        split,           # split, a tuple/list allowing concat of subsets
        backbone_type,
        round,
        use_full,  # use full dataset for training
        feat_folder,     # folder for features
        json_file,       # json file for annotations
        feat_stride,     # temporal stride of the feats
        num_frames,      # number of frames for each feat
        default_fps,     # default fps
        downsample_rate, # downsample rate for feats
        max_seq_len,     # maximum sequence length during training
        trunc_thresh,    # threshold for truncate an action segment
        crop_ratio,      # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,       # input feat dim
        num_classes,     # number of action categories
        file_prefix,     # feature file prefix if any
        file_ext,        # feature file extension if any
        force_upsampling # force to upsample to max_seq_len
    ):
        # file path
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == 'None' or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = None if crop_ratio == 'None' else crop_ratio
        self.backbone_type = backbone_type
        self.round = round
        self.data_root = '/data3/xiaodan8/FineGym'
        self.use_full = use_full

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        assert len(label_dict) == num_classes
        self.data_list = dict_db
        self.label_dict = label_dict

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'finegym',
            'tiou_thresholds': np.linspace(0.3, 0.7, 5),
            # we will mask out cliff diving
            'empty_label_ids': [],
        }
        self.transform = get_transforms(is_training, 224)

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value['annotations']:
                    label_dict[act['label']] = act['label_id']

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue
            # or does not have the feature file
            feat_file = os.path.join(self.feat_folder, self.file_prefix + key.split("_")[0] + self.file_ext)
            if not os.path.exists(feat_file):
                continue

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            with open('/data3/xiaodan8/FineGym/vid_info.json', 'r') as f:
                vid_info = json.load(f)
            duration = value['duration_frame'] / fps

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                # a fun fact of THUMOS: cliffdiving (4) is a subset of diving (7)
                # our code can now handle this corner case
                segments, labels = [], []
                for act in value['annotations']:
                    segments.append(act['segment'])
                    labels.append([label_dict[act['label']]])

                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
                locations = np.asarray(value['locations'], dtype=np.int64)
            else:
                segments = None
                labels = None
                locations = None
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels,
                         'locations' : locations
            }, )

        assert all(len(x['segments']) > 0 for x in dict_db), "Empty segments found!"
        assert all(len(x['labels']) > 0 for x in dict_db), "Empty labels found!"


        # dict_db = dict_db[:10] # for debugging, remove this line in production
        if 'train' in self.split and not self.use_full:
            video_list_by_cat = defaultdict(list)
            for v_item in dict_db:
                for anno in set(v_item['labels']):
                    video_list_by_cat[anno].append(v_item)

            # assign N annotations per category based on round number
            video_list = []
            num_anno_per_class = self.round // 99
            for k_item, v_item in video_list_by_cat.items():
                video_list.extend(circ_slice(v_item, 0, num_anno_per_class))
            dict_db = video_list
            shuffle(dict_db)
            print('uniform: ', self.split, len(video_list))


        # 1) Offline: compute dataset-level instance counts per class from dict_db
        global class_weights
        inst_counts = torch.zeros(self.num_classes, dtype=torch.long)

        for video in dict_db:
            segments = video["segments"]   # list of segments (we don't actually need them here)
            labels   = video["labels"]     # list of labels for each segment

            # assume len(segments) == len(labels)
            for seg_idx in range(len(labels)):
                seg_labels = labels[seg_idx]

                # seg_labels can be a single int (including numpy.int64) or a list of ints
                if isinstance(seg_labels, (int, np.integer)):
                    class_ids = [seg_labels]
                else:
                    class_ids = seg_labels  # assume iterable of ints

                for c in class_ids:
                    inst_counts[c] += 1
        

        # make sure it's the right length
        assert inst_counts.shape[0] == self.num_classes

        eps = 1e-6

        # convert to float
        counts_float = inst_counts.to(torch.float32)

        # handle classes that never appear in training (counts == 0)
        nonzero_mask = counts_float > 0
        if nonzero_mask.any():
            mean_nonzero = counts_float[nonzero_mask].mean()
        else:
            mean_nonzero = torch.tensor(1.0, dtype=torch.float32)

        counts_float[~nonzero_mask] = mean_nonzero

        # normalize so "typical" class has count ~ 1
        counts_float = counts_float / mean_nonzero

        # hyperparameters for weighting
        beta = 0.7           # slightly softer than 0.9
        min_weight = 1.0     # never down-weight
        max_weight = 4.0     # keep this modest
        lambda_max = 0.3     # also softer; you can tune
        n_total_samples = 20000
        stop_frac = 0.5

        x = self.round / n_total_samples
        lambda_strength = lambda_max * max(0.0, 1.0 - x / stop_frac)

        # --- rare-only weighting ---
        rare_threshold = torch.quantile(counts_float[nonzero_mask], 0.3)  # 30% percentile

        weights_raw = torch.ones_like(counts_float)
        rare_mask = counts_float < rare_threshold

        weights_raw[rare_mask] = (rare_threshold / (counts_float[rare_mask] + eps)) ** beta

        weights_raw[torch.isnan(weights_raw)] = 1.0
        weights_raw[torch.isinf(weights_raw)] = 1.0

        # clamp but NEVER below 1.0
        weights_raw = torch.clamp(weights_raw, min=min_weight, max=max_weight)

        # ---- IMPORTANT CHANGE: no normalization here ----
        # heads stay at >=1, rares >1
        class_weights = 1.0 + lambda_strength * (weights_raw - 1.0)


        return dict_db, label_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]

        # load features
        raw_stride = 16 # interval when sample video frames, see the second digit in finegym_merged_win512_int2.json
        if self.backbone_type == 'convTransformer' or self.backbone_type == 'conv':
            filename = os.path.join(self.feat_folder, self.file_prefix + video_item['id'].split("_")[0] + self.file_ext)
            feats = np.load(filename).astype(np.float32).squeeze()

            ft_idxes = [torch.floor_divide(i, self.feat_stride) for i in video_item['locations']]
            # snippet_fts = [feats[int(i)].squeeze() for i in ft_idxes]
            snippet_fts = [feats[min(int(i), len(feats) - 1)].squeeze() for i in ft_idxes]
            feats = np.stack(snippet_fts)
            # deal with downsampling (= increased feat stride)
            feats = feats[::self.downsample_rate, :]
            # T x C -> C x T
            feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))
        else:
            vid_frm = []
            for f_id in video_item['locations']:
                vid_frm.append(Image.open('/data3/xiaodan8/FineGym/RGB/'+video_item['id'].split("_")[0]+'/%07d.jpg' % f_id))
            feats = torch.stack([self.transform(frm) for frm in vid_frm])
            # deal with downsampling (= increased feat stride)
            feats = feats[::self.downsample_rate, :]
            # T x C x H x W -> C x T x H x W
            feats = torch.from_numpy(np.ascontiguousarray(feats.transpose(0,1)))

        seg_stride = self.downsample_rate * raw_stride
        feat_offset = 0.5 * self.num_frames / seg_stride

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / seg_stride - feat_offset
            )
            labels = torch.from_numpy(video_item['labels'])
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : seg_stride,
                     'feat_num_frames' : self.num_frames}

        assert len(data_dict['segments']) > 0, "Empty segments found before truncation!"

        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict_new = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
            )

            if not len(data_dict_new['segments']) > 0:
                print('Empty segments found after truncation!') # 'E3AHJ6-QS8M_79872_train'
                data_dict_new = truncate_feats(
                    data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
                )
            
            data_dict = data_dict_new

        return data_dict
