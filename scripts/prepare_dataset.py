import numpy as np
import click
import os 
from os.path import join


@click.command()
@click.option('--input_path', help='Path to feature folder that has to be prepared for training')
@click.option('--gt_path', help='Path to corresponding ground truth folder')
@click.option('--check', help='Only check dimensions', default=False, is_flag=True)
@click.option('--transpose', help='Transpose found np arrays', default=False, is_flag=True)
def run(input_path, gt_path, check, transpose):
    features = os.listdir(input_path)

    for path in features:
        feat = np.load(join(input_path, path))
        if transpose:
            print(f'Transposing {path} now with shape {feat.shape}')
            new_feature = np.transpose(feat)
            np.save(join(input_path, path), new_feature)
            continue

        # load corresponding gt
        ground_truth_path = join(gt_path, os.path.splitext(path)[0] + '.txt')
        if not os.path.exists(ground_truth_path):
            print(f'GT file {ground_truth_path} does not exist, skipping...')
            continue
        ground_truth = open(ground_truth_path, 'r').readlines()
        if ground_truth[-1][-1] != "\n": 
            ground_truth[-1] = "background\n"
        gt_len = len(ground_truth)

        difference = gt_len - feat.shape[1]
        if feat.shape[0] > feat.shape[1] and feat.shape[0] != 1024:
            print('Incorrect feature shape, is the matrix transposed yet? ', feat.shape)
            break
        if gt_len == feat.shape[1]:
            # print(f'Skip smaple {path}, dimensions already equal gt/dim {gt_len}/{feat.shape}')
            continue
        
        print(f'Comparision {path}: gt_len {gt_len}, feat_shape {feat.shape}, diff {difference}') 

        if not check and difference > 0: 
            print(f"Len difference of {difference} detected, adjusting ground truth: {ground_truth_path}")
            new_ground_truth = ground_truth[:feat.shape[1]]
            open(ground_truth_path, 'w').writelines(new_ground_truth) 
            print(f"Adjusted ground truth: {len(new_ground_truth)}/{feat.shape}")   
        elif not check and difference < 0:
            print(f"Len difference of {difference} detected, adjusting ground truth: {ground_truth_path}")
            open(ground_truth_path, 'a').write('\n')
            gt_ptr = open(ground_truth_path, 'w')
            ground_truth.extend(['background\n'] * (-1 * difference))
            new_ground_truth = [line if line != "\n" else 'background\n' for line in ground_truth]  
            gt_ptr.writelines(new_ground_truth)
            gt_ptr.close()
            print(f"Adjusted ground_truth: {len(new_ground_truth)}/{feat.shape}")
        #elif (gt_len - 1) == feat.shape[1] and check:
        #    print('Len difference of 1 detected, adjusting: ', path)
        #    new_ground_truth = ground_truth[:-1]
        #    open(ground_truth_path, 'w').writelines(new_ground_truth) 
        #    print(f"Adjusted ground truth: {len(new_ground_truth)}/{feat.shape}")


if __name__ == "__main__": 
    run()
