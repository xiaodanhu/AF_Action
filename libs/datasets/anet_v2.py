import os
import json
import numpy as np


import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from PIL import Image
from collections import defaultdict
import av
from pathlib import Path
from tqdm import tqdm
import glob
import math

from .datasets import register_dataset
from .data_utils import truncate_feats, get_transforms, truncate_video, read_video_pyav
from ..utils import remove_duplicate_annotations

@register_dataset("anet_v2")
class ActivityNetDataset(Dataset):
    def __init__(
        self,
        is_training,     # if in training mode
        split,           # split, a tuple/list allowing concat of subsets
        backbone_type,
        round,
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
        assert os.path.exists(feat_folder) or os.path.exists(json_file)
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
        self.label_dict = json.load(open('/data3/xiaodan8/Activitynet/label_dict.json'))
        self.crop_ratio = None if crop_ratio == 'None' else crop_ratio
        self.backbone_type = backbone_type
        self.round = round
        self.data_root = '/data3/xiaodan8/Activitynet'

        # load database and select the subset
        dict_db = self._load_json_db(self.json_file)
        assert len(self.label_dict) == num_classes
        self.data_list = dict_db

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'ActivityNet 1.3',
            'tiou_thresholds': np.linspace(0.5, 0.95, 10),
            'empty_label_ids': []
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

        vid_info_path = "/data3/xiaodan8/Activitynet/frame_count_raw_video.json"
        with open(vid_info_path, "r") as vid_info_file:
            vid_info = json.load(vid_info_file)

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue

            # get fps and duration
            fps = vid_info[key]['video_fps']
            num_frames = vid_info[key]['total_frames']
            duration = num_frames / fps

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                valid_acts = remove_duplicate_annotations(value['annotations'])
                num_acts = len(valid_acts)
                segments = np.zeros([num_acts, 2], dtype=np.float32)
                labels = np.zeros([num_acts, ], dtype=np.int64)
                for idx, act in enumerate(valid_acts):
                    segments[idx][0] = act['segment'][0]
                    segments[idx][1] = act['segment'][1]
                    if self.num_classes == 1:
                        labels[idx] = 0
                    else:
                        labels[idx] = self.label_dict[act['label']]
            else:
                segments = None
                labels = None
            if segments is None or len(segments) == 0:
                continue
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels
            }, )


        assert all(len(x['segments']) > 0 for x in dict_db), "Empty segments found!"
        assert all(len(x['labels']) > 0 for x in dict_db), "Empty labels found!"


        if self.is_training:
            dict_db = dict_db[:500] # for debug
        else:
            dict_db = dict_db[:100]

        ##########################
        
        for i_vid, v_data in enumerate(tqdm(dict_db, total=len(dict_db))):
            v_name = v_data['id']
            total_frames = int(v_data['fps'] * v_data['duration'])
            # Dynamically calculate stride to get self.max_seq_len frames
            if total_frames < self.max_seq_len:
                frame_indices = np.linspace(0, total_frames - 1, total_frames).astype(int)
            else:
                frame_indices = np.linspace(0, total_frames - 1, self.max_seq_len).astype(int)

            cur_vid = np.array(sorted(list(frame_indices)))

            if all([os.path.isfile(f'{self.data_root}/RGB/{v_name}'+'/%07d.jpg' % f_id) for f_id in cur_vid]):
                continue

            if not os.path.isdir(f'{self.data_root}/RGB/{v_name}'):
                Path(f'{self.data_root}/RGB/{v_name}').mkdir(parents=True, exist_ok=True)

            print("starting: ", v_name)
            container = av.open(glob.glob(os.path.join(self.data_root, 'video/v_' + v_name + '*'))[0])
            need_to_save_cur_vid = [f_id for f_id in cur_vid if not os.path.isfile(f'{self.data_root}/RGB/'+v_name+'/%07d.jpg' % f_id)]
            print("saving frames for vid: " + v_name, ", num of frames: ", len(need_to_save_cur_vid))

            batch_size = 10000  # Adjust the batch size as needed
            num_batches = math.ceil(len(need_to_save_cur_vid) / batch_size)

            for i in range(num_batches):
                batch_indices = need_to_save_cur_vid[i * batch_size:(i + 1) * batch_size]
                vid_frm = read_video_pyav(container=container, indices=batch_indices)
                # if vid_frm is None: continue
                for frame, f_id in tqdm(zip(vid_frm, batch_indices), total=len(vid_frm)):
                    im = Image.fromarray(frame)
                    im.save(f'{self.data_root}/RGB/'+v_name+'/%07d.jpg' % f_id)

        print("done")
        return dict_db

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]

        # load features
        if self.backbone_type == 'convTransformer' or self.backbone_type == 'conv':
            # case 1: convTransformer or conv to load pre-computed features
            filename = os.path.join(self.feat_folder, self.file_prefix + "_".join(video_item['id'].split("_")[:3]) + self.file_ext)
            feats = np.load(filename).astype(np.float32)
            seq_len = feats.shape[0]
            assert seq_len <= self.max_seq_len
            seq_len = self.max_seq_len
            # T x C -> C x T
            feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

            # resize the features if needed
            if (feats.shape[-1] != self.max_seq_len):
                resize_feats = F.interpolate(
                    feats.unsqueeze(0),
                    size=self.max_seq_len,
                    mode='linear',
                    align_corners=False
                )
                feats = resize_feats.squeeze(0)
        else:
            # case 2: ActionFormerWithViT or ActionFormerWithViViT to load frames
            total_frames = int(video_item['fps'] * video_item['duration'])
            # Dynamically calculate stride to get self.max_seq_len frames
            if total_frames < self.max_seq_len:
                frame_indices = np.linspace(0, total_frames - 1, total_frames).astype(int)
            else:
                frame_indices = np.linspace(0, total_frames - 1, self.max_seq_len).astype(int)
            vid_id = video_item['id']
            vid_frm = []
            for idx in frame_indices:
                frame_path = f'{self.data_root}/RGB/{vid_id}/%07d.jpg' % idx
                if os.path.exists(frame_path):
                    vid_frm.append(Image.open(frame_path))
                else:
                    # If frame missing, fill with black image
                    vid_frm.append(Image.new('RGB', (224, 224), (0, 0, 0)))
            feats = torch.stack([self.transform(frm) for frm in vid_frm])
            # T x C x H x W -> C x T x H x W
            feats = feats.permute(1, 0, 2, 3).contiguous()
            seq_len = feats.shape[1]

        
        feat_stride = video_item['duration'] * video_item['fps'] / seq_len
        # center the features
        num_frames = feat_stride
        feat_offset = 0.5 * num_frames / feat_stride

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
            )
            labels = torch.from_numpy(video_item['labels'])
            # for activity net, we have a few videos with a bunch of missing frames
            # here is a quick fix for training
            if self.is_training:
                vid_len = feats.shape[1] + feat_offset
                valid_seg_list, valid_label_list = [], []
                for seg, label in zip(segments, labels):
                    if seg[0] >= vid_len:
                        # skip an action outside of the feature map
                        continue
                    # skip an action that is mostly outside of the feature map
                    ratio = (
                        (min(seg[1].item(), vid_len) - seg[0].item())
                        / (seg[1].item() - seg[0].item())
                    )
                    if ratio >= self.trunc_thresh:
                        valid_seg_list.append(seg.clamp(max=vid_len))
                        # some weird bug here if not converting to size 1 tensor
                        valid_label_list.append(label.view(1))
                segments = torch.stack(valid_seg_list, dim=0)
                labels = torch.cat(valid_label_list)
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
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
