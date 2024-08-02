import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed
import cv2

import json
import pandas as pd
from openpyxl.utils import get_column_letter

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, arm_delay_time,
                 use_depth_image, use_robot_base):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.use_depth_image = use_depth_image
        self.arm_delay_time = arm_delay_time
        self.use_robot_base = use_robot_base
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        """
        image and qpos data at the step "start_ts"
        action sequence from the step "start_ts"
        """
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            is_compress = root.attrs['compress']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if self.use_robot_base:
                original_action_shape = (original_action_shape[0], original_action_shape[1] + 2)

            start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            if self.use_robot_base:
                qpos = np.concatenate((qpos, root['/base_action'][start_ts]), axis=0)
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            image_depth_dict = dict()
            for cam_name in self.camera_names:
                if is_compress:
                    decoded_image = root[f'/observations/images/{cam_name}'][start_ts]
                    image_dict[cam_name] = cv2.imdecode(decoded_image, 1)
                    # print(image_dict[cam_name].shape)
                    # exit(-1)
                else:
                    image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]

                if self.use_depth_image:
                    image_depth_dict[cam_name] = root[f'/observations/images_depth/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            ## HACK Note that, master states `/action` should be viewed as actions, which is also consistent with
            ## paper and source codes. But AgileX uses `/observations/qpos` as actions here. If you switch
            ## between them, do not forgt to modify the following normalization line for `action_data` and
            ## the lambda functions in the function `model_inference()` in the file `inference.py` accordingly.

            ## `/action` as actions
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # HACK, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # HACK, to make timesteps more aligned

            ## `/observations/qpos` as actions
            # actions = root['/observations/qpos'][1:]
            # actions = np.append(actions, actions[-1][np.newaxis, :], axis=0) # repeat the action at the last time step
            # start_action = min(start_ts, episode_len - 1)
            # index = max(0, start_action - self.arm_delay_time)
            # action = actions[index:]  # hack, to make timesteps more aligned
            # if self.use_robot_base:
            #     base_actions = root['/base_action'][1:] #HACK: to be consistent with root['/action'][1:]
            #     action = np.concatenate((action, base_actions[index:]), axis=1)
            # action_len = episode_len - index  # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32) # pad the array to original length
        padded_action[:action_len] = action
        action_is_pad = np.zeros(episode_len)
        action_is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        action_is_pad = torch.from_numpy(action_is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        # normalize image and change dtype to float
        image_data = image_data / 255.0
        image_depth_data = np.zeros(1, dtype=np.float32)
        if self.use_depth_image:
            all_cam_images_depth = []
            for cam_name in self.camera_names:
                all_cam_images_depth.append(image_depth_dict[cam_name])
            all_cam_images_depth = np.stack(all_cam_images_depth, axis=0)
            image_depth_data = torch.from_numpy(all_cam_images_depth)
            # image_depth_data = torch.einsum('k h w c -> k c h w', image_depth_data)
            image_depth_data = image_depth_data / 255.0

        # action_data = (action_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"] #HACK
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        # torch.set_printoptions(precision=10, sci_mode=False)
        # torch.set_printoptions(threshold=float('inf'))
        # print("qpos_data:", qpos_data[7:])
        # print("action_data:", action_data[:, 7:])

        return image_data, image_depth_data, qpos_data, action_data, action_is_pad


def get_norm_stats(dataset_dir, num_episodes, use_robot_base):
    """
    After normalization, the mean and std become 0 and 1.
    """
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
            if use_robot_base:
                qpos = np.concatenate((qpos, root['/base_action'][()]), axis=1)
                action = np.concatenate((action, root['/base_action'][()]), axis=1)
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, arm_delay_time, use_depth_image,
              use_robot_base, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action  返回均值和方差
    norm_stats = get_norm_stats(dataset_dir, num_episodes, use_robot_base)

    # construct dataset and dataloader 归一化处理  结构化处理数据
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, arm_delay_time,
                                    use_depth_image, use_robot_base)

    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, arm_delay_time,
                                  use_depth_image, use_robot_base)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True,
                                  num_workers=1, prefetch_factor=1)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1,
                                prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def store_params(args, ckpt_folder_name):
    param_record = {"policy_class": args.policy_class,
                "batch_size": args.batch_size, 
                "lr": args.lr, 
                "chunk_size": args.chunk_size, 
                "num_epochs": args.num_epochs, 
                "num_episodes": args.num_episodes,
                "kl_weight": args.kl_weight,
                "hidden_dim": args.hidden_dim,
                "dim_feedforward": args.dim_feedforward,
                "seed": args.seed
                }
    # store important parameters into .json file
    param_file = os.path.join(args.ckpt_dir, args.task_name+".json")
    if not os.path.exists(param_file):
        with open(param_file, 'w') as f:
            json.dump({}, f)
    with open(param_file, 'r') as f:
        records = json.load(f)
    records[ckpt_folder_name] = param_record
    with open(param_file, 'w') as f:
        json.dump(records, f, indent=4)
    # visualize stored parameters in .xlsx file
    param_table = os.path.join(args.ckpt_dir, args.task_name+".xlsx")
    with pd.ExcelWriter(param_table, engine='openpyxl') as writer:
        df = pd.DataFrame.from_dict(records, orient='index')
        df.to_excel(writer)

        workbook = writer.book
        worksheet = writer.sheets['Sheet1']

        # adjust column width
        for column_cells in worksheet.columns:
            length = 1.5*max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
            worksheet.column_dimensions[get_column_letter(column_cells[0].column)].width = length