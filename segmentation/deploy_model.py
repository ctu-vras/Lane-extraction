#from dataloader.deployment_dataset import DeploymentDataset
from segmentation.dataloader.deployment_dataset import DeploymentDataset
#from builder import model_builder
from segmentation.builder import model_builder
#from config.config import load_config_data
from segmentation.config.config import load_config_data
#from utils.load_save_util import load_checkpoint
from segmentation.utils.load_save_util import load_checkpoint
#from dataloader.dataset_semantickitti import get_model_class, collate_fn_BEV
from segmentation.dataloader.dataset_semantickitti import get_model_class, collate_fn_BEV
import numpy as np
import torch
from tqdm import tqdm
import os

# TODO confidence not yet working

def get_dataloader(point_cloud: np.ndarray, partition_size: float, voxel_size: float, configs):
    pt_dataset = DeploymentDataset(
        point_cloud=point_cloud,
        partition_size=partition_size,
        voxel_size=voxel_size
    )

    masks, grid_min_coords, grid_indices, downsamled_point_cloud = pt_dataset.generate_split()

    dataset = get_model_class('voxel_dataset')(
        pt_dataset,
        grid_size=configs['model_params']['output_shape'],
        flip_aug=False,
        fixed_volume_space=configs['dataset_params']['fixed_volume_space'],
        max_volume_space=configs['dataset_params']['max_volume_space'],
        min_volume_space=configs['dataset_params']['min_volume_space'],
        ignore_label=configs['dataset_params']["ignore_label"],
        rotate_aug=False,
        return_test=True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        collate_fn=collate_fn_BEV,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=1,
        num_workers=3
    )

    return dataloader, masks, grid_min_coords, grid_indices, downsamled_point_cloud

def upsample_mask(full_point_cloud: np.ndarray, downsampled_mask: np.ndarray, downsampled_indices: np.ndarray, min_coords: np.ndarray, voxel_size: float):
    def get_mask(subset: np.ndarray, source: np.ndarray):
        b = subset
        a = source
        baseval = np.max([a.max(), b.max()]) + 1
        n_cols = a.shape[1]
        a = a * baseval ** np.array(range(n_cols))
        b = b * baseval ** np.array(range(n_cols))
        c = np.isin(np.sum(a, axis=1), np.sum(b, axis=1))
        return c

    full_grid_indices = np.floor((full_point_cloud[:, :3] - min_coords.reshape(-1, 1).T) / voxel_size).astype(np.int64)
    valid_downsampled_indices = downsampled_indices[downsampled_mask.astype(bool)]
    full_mask = get_mask(np.unique(valid_downsampled_indices, axis=0), full_grid_indices)
    return full_mask

def run_model(model, dataloader, model_config, pipeline_config):

    def label_pts(grid_ten, vox_label):
        point_labels_ten = vox_label[0, :]
        grid = grid_ten[0].cpu()
        pt_lab_lin = point_labels_ten.reshape(-1)
        grid_lin = np.ravel_multi_index(grid.numpy().T, model_config['output_shape'])
        out_labels = pt_lab_lin[torch.from_numpy(grid_lin)]
        labels = out_labels.reshape(-1, 1)
        return labels

    def map_outputs_to_pts(grid_indices: np.ndarray, outputs: np.ndarray):
        pt_vox_outputs = outputs[0, :, grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]]
        return pt_vox_outputs

    print(f'Getting model output -- {len(dataloader)}')

    if pipeline_config['MODEL_SAVE_DEBUG'] and not os.path.isdir('./debug_patches'):
        os.makedirs('./debug_patches/')
        print('saving debug outputs to: ./debug_patches')

    outputs_dict = dict()
    model.eval()
    pbar = tqdm(total=len(dataloader))
    for i_iter_train, (
            xyzil, train_vox_label, train_grid, _, train_pt_fea, ref_st_idx, ref_end_idx, lcw) in enumerate(
        dataloader):

        index = int(xyzil[0, 0, -1].item())
        pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pipeline_config['CUDA_CARD']) for i in train_pt_fea]
        vox_ten = [torch.from_numpy(i).to(pipeline_config['CUDA_CARD']) for i in train_grid]
        label_tensor = train_vox_label.type(torch.LongTensor).to(pipeline_config['CUDA_CARD'])
        grid_ten = [torch.from_numpy(i).to(pipeline_config['CUDA_CARD']) for i in train_grid]

        outputs = model(pt_fea_ten, vox_ten, 1)

        if torch.isnan(outputs).any():
            print('outputs NaN')

        inp = label_tensor.size(0)
        outputs = outputs[:inp, :, :, :, :]
        mapped_outputs = map_outputs_to_pts(train_grid[0], outputs.clone().cpu().numpy())
        predict_labels = torch.argmax(outputs, dim=1)
        labels = label_pts(grid_ten, predict_labels)
        outputs_dict[index] = [mapped_outputs, xyzil, labels.cpu().numpy()]

        if pipeline_config['MODEL_SAVE_DEBUG']:
            patch_path = os.path.join('./debug_patches', f'patch_{index}.npz')
            patch_pts = np.concatenate([xyzil[0, :, :], labels.cpu().numpy().reshape(-1, 1)], axis=1)
            np.savez(patch_path, data=patch_pts)

        pbar.update(1)
    pbar.close()
    return outputs_dict

def softmax_outputs(output_logits: np.ndarray):
    softmaxed_logits = np.exp(output_logits) / np.sum(np.exp(output_logits), axis=1, keepdims=True)
    return softmaxed_logits

def get_labels(outputs_dict: dict, downsampled_point_cloud: np.ndarray, masks: list, pipeline_config):

    # TODO implement overlap decision heuristics - use softmax probabilities
    label_dim = np.zeros_like(downsampled_point_cloud[:, 0])
    confidences = -np.ones_like(label_dim)

    for index in range(len(masks)):
        logits, _, labels = outputs_dict[index]
        mask = masks[index]
        probablities = softmax_outputs(logits)
        prob_diffs = np.abs(probablities[:, 0] - probablities[:, 1])

        # TODO fix
        if pipeline_config['MODEL_SELECTION_STRATEGY'] == 'last':
            label_dim[mask] = labels.reshape(-1)
        elif pipeline_config['MODEL_SELECTION_STRATEGY'] == 'or':
            label_dim[mask] = np.logical_or(label_dim[mask].astype(bool), labels.astype(bool))
        elif pipeline_config['MODEL_SELECTION_STRATEGY'] == 'confidence':
            prob_diff_mask = prob_diffs > confidences[mask]
            label_dim[mask][prob_diff_mask] = labels.reshape(-1)[prob_diff_mask]
        else:
            print('unknown strategy')

        #prob_diff_mask = prob_diffs > confidences[mask]
        #label_dim[mask][prob_diff_mask] = labels.reshape(-1)[prob_diff_mask]
        confidences[mask] = prob_diffs.reshape(-1)

    labeled_point_cloud = np.concatenate([downsampled_point_cloud, label_dim.reshape(-1, 1), confidences.reshape(-1, 1)], axis=1)
    return labeled_point_cloud

def deploy_model(point_cloud: str, pipeline_config):

    # load configuration
    configs = load_config_data(pipeline_config['MODEL_CONFIG_PATH'])
    model_config = configs['model_params']
    train_hypers = configs['train_params']
    model_path = train_hypers['model_load_path']

    # load model
    model = model_builder.build(model_config)
    model = model.to(pipeline_config['CUDA_CARD'])
    model = load_checkpoint(model_path, model, map_location=pipeline_config['CUDA_CARD'][-1])

    # load dataset
    dataloader, masks, grid_min_coords, grid_indices, downsampled_point_cloud = get_dataloader(
        point_cloud=point_cloud,
        partition_size=pipeline_config['MODEL_PARTITION_SIZE'],
        voxel_size=pipeline_config['MODEL_VOXEL_SIZE'],
        configs=configs
    )

    outputs_dict = run_model(model, dataloader, model_config, pipeline_config)

    print('Labeling outputs')
    labeled_downsample = get_labels(outputs_dict, downsampled_point_cloud, masks, pipeline_config)

    if pipeline_config['MODEL_RETURN_UPSAMPLED']:
        labels = labeled_downsample[:, -2]
        full_labels = upsample_mask(
            point_cloud,
            labels,
            grid_indices,
            grid_min_coords,
            pipeline_config['MODEL_VOXEL_SIZE']
        )

        return full_labels
    else:
        return labeled_downsample

