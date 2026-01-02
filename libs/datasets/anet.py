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

@register_dataset("anet")
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

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue

            # get fps if available
            if 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            duration = value['duration_frame'] / fps

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                valid_acts = remove_duplicate_annotations(value['annotations'])
                num_acts = len(valid_acts)
                segments = np.zeros([num_acts, 2], dtype=np.float32)
                labels = np.zeros([num_acts, ], dtype=np.int64)
                for idx, act in enumerate(valid_acts):
                    segments[idx][0] = act['segment'][0]
                    segments[idx][1] = act['segment'][1]
                    labels[idx] = self.label_dict[act['label']]
                    sample_stride = act['sample_stride']
                locations = np.sort(np.asarray(value['locations'], dtype=np.int64))
            else:
                segments = None
                labels = None
                locations = None
            if segments is None or len(segments) == 0:
                continue
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels,
                         'locations' : locations,
                         'sample_stride' : sample_stride
            }, )


        assert all(len(x['segments']) > 0 for x in dict_db), "Empty segments found!"
        assert all(len(x['labels']) > 0 for x in dict_db), "Empty labels found!"


        # if self.is_training:
        #     dict_db = dict_db[:500] # for debug
        # else:
        #     dict_db = dict_db[:100]

        ##########################
        
        # v_dict = defaultdict(set)
        # for v in dict_db:
        #     vid = "_".join(v['id'].split("_")[:-1])
        #     v_dict[vid].update(v['locations'])

        # for i_vid, vid in enumerate(tqdm(list(v_dict.keys())[:], total=len(v_dict))):
        #     # if not (vid == 'GCyW1jPfWdg' or vid == 'XL8N3BEWdho' or vid == 'hMeTZBe6UBk' or vid == 'jR0OFsCYcD4' or vid == 'x1owr0LEc3M'):
        #     #     continue

        #     cur_vid = np.array(sorted(list(v_dict[vid])))

        #     if all([os.path.isfile(f'{self.data_root}/RGB/{vid}'+'/%07d.jpg' % f_id) for f_id in cur_vid]):
        #         continue

        #     if not os.path.isdir(f'{self.data_root}/RGB/{vid}'):
        #         Path(f'{self.data_root}/RGB/{vid}').mkdir(parents=True, exist_ok=True)

        #     print("starting: ", vid)
        #     container = av.open(glob.glob(os.path.join(self.data_root, 'video/v_' + vid + '*'))[0])
        #     need_to_save_cur_vid = [f_id for f_id in cur_vid if not os.path.isfile(f'{self.data_root}/RGB/'+vid+'/%07d.jpg' % f_id)]
        #     print("saving frames for vid: " + vid, ", num of frames: ", len(need_to_save_cur_vid))

        #     batch_size = 10000  # Adjust the batch size as needed
        #     num_batches = math.ceil(len(need_to_save_cur_vid) / batch_size)

        #     for i in range(num_batches):
        #         batch_indices = need_to_save_cur_vid[i * batch_size:(i + 1) * batch_size]
        #         vid_frm = read_video_pyav(container=container, indices=batch_indices)
        #         # if vid_frm is None: continue
        #         for frame, f_id in tqdm(zip(vid_frm, batch_indices), total=len(vid_frm)):
        #             im = Image.fromarray(frame)
        #             im.save(f'{self.data_root}/RGB/'+vid+'/%07d.jpg' % f_id)
        
        # # print progress when every 10% of the v_dict is done 
        # if i_vid % (len(v_dict) // 10) == 0:
        #     print(f"Progress: {i_vid / len(v_dict) * 100:.2f}%")

        return dict_db

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]

        # load features
        raw_stride = video_item['sample_stride'] # interval when sample video frames, see the second digit in finegym_merged_win128_int4.json
        if self.backbone_type == 'convTransformer' or self.backbone_type == 'conv':
            filename = os.path.join(self.feat_folder, self.file_prefix + "_".join(video_item['id'].split("_")[:3]) + self.file_ext)
            feats = np.load(filename).astype(np.float32).squeeze()

            ft_idxes = [torch.floor_divide(i, self.feat_stride) for i in video_item['locations']]
            snippet_fts = [feats[min(int(i), len(feats) - 1)].squeeze() for i in ft_idxes]
            feats = np.stack(snippet_fts)
            # deal with downsampling (= increased feat stride)
            feats = feats[::self.downsample_rate, :]
            # T x C -> C x T
            feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))
        else:
            vid_frm = []
            T = len(video_item['locations'])
            last_valid_idx = 0
            for cur_idx in range(T):
                f_id = video_item['locations'][cur_idx]
                # check if all frames are available
                if not os.path.isfile(f'{self.data_root}/RGB/'+"_".join(video_item['id'].split("_")[:-1])+'/%07d.jpg' % f_id):
                    # use the last valid frame
                    frame_to_use = video_item['locations'][last_valid_idx]
                else:
                    frame_to_use = f_id
                    last_valid_idx = cur_idx
                vid_frm.append(Image.open(f'{self.data_root}/RGB/'+"_".join(video_item['id'].split("_")[:-1])+'/%07d.jpg' % frame_to_use))
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
