"""
autoregression_test.py — Inference satellite IR brightness-temperature data with AI model DaYu.

SPDX-License-Identifier: Apache-2.0

"""

import gc
import argparse
import numpy as np
import pandas as pd
import torch
import datetime
import os
import glob
from datetime import datetime, timedelta
from datasets import data_utils
from models import build_model
from utils import str2bool


def load_file(root):

    " here is your custom load file codes, for example, np.load(root)"
    concatenated_array = np.load(root)

    concatenated_tensor = data_utils.read_img(concatenated_array).unsqueeze(0)  # to tensor [C H W]
    concatenated_tensor[torch.isnan(concatenated_tensor)] = 0.0
    concatenated_tensor[concatenated_tensor < 0] = 0.0

    return concatenated_tensor


def encode_path(path):
    time = os.path.splitext(os.path.basename(path))[0]
    time = time.split('_')[2] + time.split('_')[3]
    time_format = '%Y%m%d%H%M'
    time = pd.to_datetime(time, format=time_format)
    init_time = np.array([time])

    hours = np.array([pd.Timedelta(minutes=t * 30) for t in [0, 1, 2]])
    times = init_time[:, None] + hours[None]
    times = [pd.Period(t, 'min') for t in times.reshape(-1)]
    times_numpy = [(p.day_of_year / 366, p.hour / 24, p.minute / 60) for p in times]
    temb = torch.from_numpy(np.array(times_numpy, dtype=np.float32))
    temb = torch.cat([temb.sin(), temb.cos()], dim=-1)
    time_tensor = temb.reshape(1, -1)

    return time_tensor


def create_time_coding_list(data_path):
    encoded_paths = [encode_path(path) for path in data_path]

    return encoded_paths


def get_args_parser():  # optional args for extending
    parser = argparse.ArgumentParser('Geo-test', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Per GPU batch size')

    # Model parameters
    parser.add_argument('--model', default='base_model', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=2400, type=int,
                        help='image input size')
    parser.add_argument('--drop_path', type=float, default=0., metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Dataset parameters
    parser.add_argument('--data_path', default='your_data_path', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='your_model_saved_path',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)  # torch 2.0 version
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser


@torch.no_grad()
def auto_regression(args):
    print(args)
    device = torch.device(args.device)
    cascade_mode = False
    path_1 = "your model weights path 1"

    if cascade_mode:
        ckpt_1_path = path_1

        path_2 = "your model weights path 2"
        ckpt_2_path = path_2
    else:
        ckpt_1_path = path_1
        ckpt_2_path = path_1

    # define the model
    model_1 = build_model(args)
    model_2 = build_model(args)

    checkpoint_1 = torch.load(ckpt_1_path, map_location='cpu')
    checkpoint_2 = torch.load(ckpt_2_path, map_location='cpu')

    print("Load first checkpoint from: %s" % ckpt_1_path)
    print("Load second checkpoint from: %s" % ckpt_2_path)

    model_1.to(device)
    model_2.to(device)

    model_without_ddp_1 = model_1  # shallow copy
    model_without_ddp_2 = model_2  # shallow copy
    n_parameters = sum(p.numel() for p in model_1.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp_1))
    print('number of params:', n_parameters)

    model_without_ddp_1.load_state_dict(checkpoint_1['model'])
    model_without_ddp_2.load_state_dict(checkpoint_2['model'])

    state_dict1 = model_without_ddp_1.state_dict()
    state_dict2 = model_without_ddp_2.state_dict()

    are_equal = True
    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            print(f"Parameter {key} is different.")
            are_equal = False

    if are_equal:
        print("The two checkpoints have identical parameters.")
    else:
        print("The two checkpoints have different parameters.")

    inference(args, model_without_ddp_1, model_without_ddp_2, device)


@torch.no_grad()
def inference(args, model_1, model_2, device):
    max_val = 340.0
    min_val = 160.0

    root_dir = 'your data path'

    folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))][:]
    # switch to evaluation mode
    model_1.eval()
    model_2.eval()

    step_num = 12  # one step stands for 0.5 h
    extension = '.nc'  # check your file type
    # extension = '.npy'

    for folder in folders:
        data_list = []

        files = glob.glob(os.path.join(root_dir, folder, f'*{extension}'))
        if len(files) in [0, 1]:  # len(files) need to be 2
            print(f'{folder} is invalid')
            continue

        # analyse folder name, get the time for check and encode
        try:
            folder_date = datetime.strptime(folder, '%Y_%m_%d_%H_%M')
        except ValueError:
            print(f"Invalid folder name format: {folder}")
            continue

        # calculate current time and half an hour before name
        same_time_file = folder_date.strftime('your_file_path_concluding_%Y%m%d_%H%M_*')

        half_hour_before_file = (folder_date - timedelta(minutes=30)).strftime(
            'your_file_path_concluding_%Y%m%d_%H%M_*')

        # first two files
        data_list.append(os.path.join(root_dir, folder, half_hour_before_file))
        data_list.append(os.path.join(root_dir, folder, same_time_file))

        # files list
        current_date = folder_date
        for i in range(step_num):
            current_date = current_date + timedelta(minutes=30)
            current_time_file = current_date.strftime('your_file_path_concluding_%Y%m%d_%H%M_*')

            data_list.append(os.path.join(root_dir, folder, current_time_file))

        time_tensors = create_time_coding_list(data_list)

        first_file_name = data_list[0]
        second_file_name = data_list[1]

        frame_1 = load_file(first_file_name)
        frame_2 = load_file(second_file_name)

        input = torch.cat((frame_1, frame_2), dim=1)  # tensor [1 2C H W]

        time_tensor_cpu = time_tensors[0].unsqueeze(0)

        input = input.to(device, non_blocking=True)

        # inference part
        for j in range(1, step_num):
            save_forecasted_final_name = data_list[j + 1][:-3] + '.npy'
            time_tensor = time_tensor_cpu.to(device, non_blocking=True)

            if j <= step_num / 2:
                print(f"using model 1 for {j}th autoregression. ")
                new_diff = model_1(input, time_tensor)
            else:
                print(f"using model 2 for {j}th autoregression. ")
                new_diff = model_2(input, time_tensor)

            new_data = input[:, 8:] + new_diff  # 8 is the channels' count

            output_temp = new_data * (max_val - min_val) + min_val
            output_numpy = output_temp.permute(0, 2, 3, 1).squeeze().data.cpu().numpy()

            np.save(save_forecasted_final_name, output_numpy)

            new_frame_1 = input[:, 8:]
            input = torch.cat((new_frame_1, new_data), dim=1)

            time_tensor_cpu = time_tensors[j].unsqueeze(0)
            del new_frame_1, new_data
            gc.collect()
            torch.cuda.empty_cache()

            if j == step_num - 1:
                save_forecasted_final_name = data_list[j + 2][:-3] + '.npy'

                time_tensor = time_tensor_cpu.to(device, non_blocking=True)
                output_diff = model_2(input, time_tensor)
                output = output_diff + input[:, 8:]
                output = output * (max_val - min_val) + min_val
                output_final_numpy = output.permute(0, 2, 3, 1).squeeze().data.cpu().numpy()

                np.save(save_forecasted_final_name, output_final_numpy)

        print(f"************* {folder} forecast is over! *************")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('The Prepare script', parents=[get_args_parser()])
    args = parser.parse_args()

    auto_regression(args)
