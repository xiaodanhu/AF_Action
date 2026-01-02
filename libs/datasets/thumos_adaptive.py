'''
Updated confusion matrix calculation code: for each prediction, find the best matching ground truth segment based on IoU. Skip the ground truth segment if it has already been matched to another prediction.
Implemented video hardness sampling for training data, where videos with more confusing annotations are sampled more frequently.
fixed error that sort order is wrong; only select new samples that are not in the used_vids_dict
noise filerting
'''
import os
import json
import numpy as np
from random import shuffle
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from PIL import Image

from .datasets import register_dataset
from .data_utils import truncate_feats, get_transforms, truncate_video, circ_slice, build_confusion_matrix
from collections import defaultdict
import math
from scipy.special import softmax


def update_noise_and_filter(scored, cls_idx, thresh, noise_counts, noise_rounds, filter_out=True):
    """
    Update noise counts for a list of (hardness, video) tuples.
    If filter_out is True, return only those with cnt < noise_rounds.
    If filter_out is False, return all videos (for unused vids).
    """
    selected = []
    filtered = []
    for h, v in scored:
        vid = v['id']
        cnt = noise_counts.get(cls_idx, {}).get(vid, 0)
        cnt = cnt + 1 if h >= thresh else 0
        noise_counts.setdefault(cls_idx, {})[vid] = cnt
        if filter_out and cnt >= noise_rounds:
            filtered.append(vid)
        else:
            selected.append((h, v))
    return selected, filtered


def compute_video_hardness(
    logits_dict: dict,
    cls_idx: int,
    strategy: str = 'mean'
) -> float:
    """
    Compute a *hardness* score for class `cls_idx`:
      - uses logits_dict[cls_idx] → np.array [T_feat, C]
      - higher score = more confusing (model more often favors OTHER classes)
    Args:
      logits_dict: dict[int → np.array], each [T_feat, C]
      cls_idx:     target class index
      strategy:    'min' | 'mean' | 'topk_K' (e.g. 'topk_5')
    Returns:
      float hardness (larger ⇒ more confusing)
    """
    if cls_idx not in logits_dict:
        return 0.0  # no data ⇒ treat as “easy”

    # 1) to probabilities
    logits = logits_dict[cls_idx]         # [T_feat, C]
    probs  = softmax(logits, axis=1)

    # 2) per-frame margin = p_true - max_other
    p_true = probs[:, cls_idx]
    tmp    = probs.copy()
    tmp[:, cls_idx] = -np.inf
    p_other = tmp.max(axis=1)
    margins = p_true - p_other           # [T_feat]

    if margins.size == 0:
        return 0.0

    # 3) invert & aggregate
    if strategy == 'min':
        # use the *smallest* margin (the worst), invert it
        return float(-np.min(margins))
    elif strategy == 'mean':
        return float(-np.mean(margins))
    elif strategy.startswith('topk_'):
        k = int(strategy.split('_')[1])
        smallest = np.partition(margins, k-1)[:k]
        return float(-np.mean(smallest))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
@register_dataset("thumos_adaptive")
class THUMOS14Dataset(Dataset):
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
        self.data_root = '/data3/xiaodan8/thumos'
        self.use_full = use_full

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        assert len(label_dict) == num_classes
        self.data_list = dict_db
        self.label_dict = label_dict

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'thumos14',
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
            self.label_dict = label_dict

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue
            # or does not have the feature file
            feat_file = os.path.join(self.feat_folder, self.file_prefix + "_".join(key.split("_")[:3]) + self.file_ext)
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



        if 'train' in self.split and not self.use_full:
            num_classes_fixed = self.num_classes
            per_round_num = 300
            # 1) load per-video logits from last round
            prev_round = self.round - per_round_num
            logits_path = f'/data3/xiaodan8/actionformer4_1/output/thumos14/logits_adaptive_{prev_round}.pt'
            video_logits = torch.load(logits_path)  # dict: vid -> torch.Tensor [T_feat, C]
            # ensure numpy for slicing
            video_logits = {k: {lbl: np.stack(lst) for lbl, lst in v.items()} for k, v in video_logits.items()}

            # 2) build confusion‐weighted video groups
            with open(f'/data3/xiaodan8/actionformer4_1/output/thumos14/pred_vit_adaptive_{prev_round}.json') as f:
                pred = json.load(f)
            with open('/data3/xiaodan8/actionformer4_1/thumos_test_check.json') as f:
                gt_test = json.load(f)

            cm = build_confusion_matrix(pred, gt_test['database'], self.label_dict)
            # confuse_score = ((cm.sum(axis=1) - np.diag(cm)) / (cm.sum() - cm.trace())).tolist()
            confuse_score = (cm.sum(axis=1) - np.diag(cm)) / (cm.sum(axis=1) + 1e-6)
            confuse_score = (confuse_score / confuse_score.sum()).tolist()

            global CONFUSE_SCORE
            CONFUSE_SCORE = confuse_score

            # 3) group videos by GT class
            video_list_by_cat = defaultdict(list)
            for v_item in dict_db:
                for anno in set(v_item['labels']):
                    video_list_by_cat[anno].append(v_item)
            

            ## 4.3) adaptive margin sampling: select per-class hardest videos
            video_list = []
            used_vids_dict = {}
            if not os.path.exists('/data3/xiaodan8/actionformer4_1/output/thumos14/used_vids_dict.pt'):
                # if used_vids file not exists, use the first 1000 videos in the first round
                num_anno_per_class = 500 // num_classes_fixed
                used_vids = {cls_idx: [v['id'] for v in video_list_by_cat[cls_idx][:num_anno_per_class]].copy() for cls_idx in range(self.num_classes)}
                noise_counts = {c: {} for c in range(self.num_classes)}
                used_vids_dict[str(prev_round)] = {'accumulated': used_vids.copy(), 'new': used_vids.copy(), 'noise_counts': noise_counts.copy()}
            else:
                with open('/data3/xiaodan8/actionformer4_1/output/thumos14/used_vids_dict.pt', 'r') as f:
                    used_vids_dict = json.load(f)
                used_vids = {int(k): v.copy() for k, v in used_vids_dict[str(prev_round)]['accumulated'].items()}
                noise_counts = {int(k): v.copy() for k, v in used_vids_dict[str(prev_round)]['noise_counts'].items()}
            
            selected = {}
            # parameters for noise filtering
            noise_rounds = 3
            for cls_idx, vids in video_list_by_cat.items():
                new_quota = math.ceil(confuse_score[cls_idx] * per_round_num)
                used_vid_ids = set(used_vids[cls_idx])
                noise_vid_ids = [k for k,v in noise_counts[cls_idx].items() if v >= noise_rounds]
                unused_vids = [v for v in vids if v['id'] not in used_vid_ids and v['id'] not in noise_vid_ids]
                used_vid_objs = [v for v in vids if v['id'] in used_vid_ids and v['id'] not in noise_vid_ids]

                # Compute hardness for both used and unused
                scored_unused = [(compute_video_hardness(video_logits.get(v['id']), cls_idx, strategy='mean'), v) for v in unused_vids]
                scored_used = [(compute_video_hardness(video_logits.get(v['id']), cls_idx, strategy='mean'), v) for v in used_vid_objs]

                # Combine for threshold calculation
                all_scores = [h for h, _ in scored_unused] + [h for h, _ in scored_used]
                thresh = np.percentile(all_scores, 95) if all_scores else float('inf')

                # Update noise counts and filter
                # For unused: just update noise count, always include (filter_out=False)
                selected_unused, _ = update_noise_and_filter(scored_unused, cls_idx, thresh, noise_counts, noise_rounds, filter_out=False)
                # For used: update noise count, filter out if cnt >= noise_rounds
                selected_used, filt_used = update_noise_and_filter(scored_used, cls_idx, thresh, noise_counts, noise_rounds, filter_out=True)

                # Select top unused vids within quota
                selected_unused.sort(key=lambda x: x[0], reverse=True)
                selected_unused = [v for _, v in selected_unused[:new_quota]]
                selected_used = [v for _, v in selected_used]

                selected[cls_idx] = {'used': selected_used, 'unused': selected_unused, 'filtered': filt_used}
                video_list.extend(selected_unused)
                video_list.extend(selected_used)
            # save used videos for next round
            used_vids_dict[str(self.round)] = {
                'accumulated': {cls_idx: [v['id'] for v in selected[cls_idx]['used'] + selected[cls_idx]['unused']].copy() for cls_idx in range(self.num_classes)},
                'new': {cls_idx: [v['id'] for v in selected[cls_idx]['unused']].copy() for cls_idx in range(self.num_classes)},
                'noise_counts': {k: v.copy() for k, v in noise_counts.items()},
                'filtered': {cls_idx: selected[cls_idx]['filtered'].copy() for cls_idx in range(self.num_classes)}
            }
            with open('/data3/xiaodan8/actionformer4_1/output/thumos14/used_vids_dict.pt', 'w') as f:
                json.dump(used_vids_dict, f)
            shuffle(video_list)
            dict_db = tuple(video_list)
            print('adaptive: ', self.split, len(video_list))

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
        n_total_samples = 2306
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
        raw_stride = 4 # interval when sample video frames, see the second digit in finegym_merged_win512_int2.json
        if self.backbone_type == 'convTransformer' or self.backbone_type == 'conv':
            filename = os.path.join(self.feat_folder, self.file_prefix + "_".join(video_item['id'].split("_")[:3]) + self.file_ext)
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
                vid_frm.append(Image.open('/data3/xiaodan8/thumos/RGB/'+"_".join(video_item['id'].split("_")[:3])+'/%07d.jpg' % f_id))
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
