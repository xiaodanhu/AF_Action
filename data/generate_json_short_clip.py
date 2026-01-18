'''
convert groundtruth from csv to json
'''

import numpy as np
import csv, json
from collections import defaultdict
import os.path
import warnings
warnings.filterwarnings("ignore")

# mode = 'test'

window_size = 32 # 64 1800 450 1024 512
interval = 16 # 16 2

def second2frame(second, fps):
    frame = int(second * fps)
    return frame

# get label dict
action_label_path = "/data3/xiaodan8/FineGym/gym99_categories.txt"
id2actionlabel = defaultdict()
with open(action_label_path, 'r') as f:
    for line in f:
        line = line.strip().split("; ")
        act_id = int(line[0].strip("Clabel:   "))
        action_name = line[3].strip()
        id2actionlabel.update({act_id: action_name})

with open("/data3/xiaodan8/FineGym/finegym_annotation_info_v1.1.json", 'r') as f:
    data = json.load(f)
with open('/data3/xiaodan8/FineGym/vid_info.json', 'r') as f:
    vid_info = json.load(f)

anno_data = defaultdict(dict)
for mode in ['train','test']:
    anno_label_path = f"/data3/xiaodan8/FineGym/gym99_{mode}.txt"
    with open(anno_label_path, 'r') as f:
        for line in f:
            line, action_id = line.strip().split(" ")
            action_id = int(action_id)
            line = line.strip().split("_")
            vname = line[0]
            activity_key = "_".join(line[1:4])
            action_key = "_".join(line[4:7])

            if f'{vname}_{mode}' not in anno_data:
                anno_data[f'{vname}_{mode}']['duration_frame'] = vid_info[vname]['duration']
                anno_data[f'{vname}_{mode}']["annotations"] = []
                anno_data[f'{vname}_{mode}']["fps"] = vid_info[vname]['fps']

            # load time stamp data
            activity_data = data[vname][activity_key]
            action_time = [x for xs in activity_data['segments'][action_key]['timestamps'] for x in xs]
            action_time = [activity_data['timestamps'][0][0]+i for i in [action_time[0], action_time[-1]]]
            action_time_frm = [second2frame(float(x), vid_info[vname]['fps']) for x in action_time]
            anno_data[f'{vname}_{mode}']["annotations"].append({'label': id2actionlabel[action_id], 'segment': action_time, 'segment(frames)': action_time_frm, 'label_id': action_id})

video_list = defaultdict(dict)
for vid_mode, annotations in anno_data.items():
    vid, mode = ("").join(vid_mode.split("_")[:-1]), vid_mode.split("_")[-1]

    num_frames = int(annotations['duration_frame'])
    fps = annotations['fps']
    frames = np.array(range(0, num_frames, interval))

    seq_len = len(frames)
    if mode == 'test':
        overlap_ratio = 1
    else:
        overlap_ratio = 2
    stride = window_size // overlap_ratio
    ws_starts = [
        i * stride
        for i in range((seq_len // window_size - 1) * overlap_ratio + 1)
    ]
    ws_starts.append(seq_len - window_size)

    for ws in ws_starts:
        locations = frames[ws:ws + window_size]
        gt = []
        for idx in range(len(annotations["annotations"])):
            anno = annotations["annotations"][idx]
            if anno["segment(frames)"][0] >= locations[0] and anno["segment(frames)"][1] <= locations[-1]:
                gt.append(
                    {
                        "label": anno["label"],
                        "segment": (anno["segment"] - locations[0] / fps).tolist(),
                        "segment(frames)": (anno["segment(frames)"] - locations[0]).tolist(),
                        "label_id": anno["label_id"]
                    }
                )
        # if annotations['subset'] == 'test' and len(gt) == 0:
        #     video_list[f'vid_{ws}']['annotations'].append({'label': '', 'segment': action_time, 'segment(frames)': action_time_frm, 'label_id': action_id})
        if len(gt) > 0:
            video_list[f'{vid}_{ws}_{mode}'] = {'duration_frame': window_size * interval, 'subset': mode, 'fps': fps, 'annotations': gt, "locations": locations.tolist()}

ground_truth = {'database': video_list}
with open(f'/data3/xiaodan8/FineGym/finegym_merged_win{window_size}_int{interval}.json', 'w') as f:
    json.dump(ground_truth, f)
# with open(f'/data3/xiaodan8/actionformer3/finegym_test_check.json', 'w') as f:
#     json.dump(ground_truth, f)