import os
import click
import tqdm


def read_ground_truth(file_path):
    with open(file_path, "r") as f:
        data = f.read().split("\n")
        if data[-1] == "":
            data = data[:-1]
    return data


def write_ground_truth(file_path, data):
    with open(file_path, "w") as f:
        f.write("\n".join(data))


def sample_with_float_step(values, n_samples):
    if n_samples == 1:
        return [values[0]]

    step = (len(values) - 1) / (n_samples - 1)
    indices = [round(i * step) for i in range(n_samples)]
    return [values[i] for i in indices]


@click.command()
@click.argument("input_dir")
@click.argument("output_dir")
@click.option("--source_fps", type=int, default=25, help="Source frames per second")
@click.option("--target_fps", type=int, default=15, help="Target frames per second")
def downsample_gt_cataract101(input_dir, output_dir, source_fps, target_fps):
    gts = os.listdir(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    for gt in tqdm.tqdm(gts):
        ground_truth = read_ground_truth(os.path.join(input_dir, gt))
        n_source_frames = len(ground_truth)
        n_target_frames = int(round(n_source_frames * target_fps / source_fps)) + 2
        downsampled_ground_truth = sample_with_float_step(ground_truth, n_target_frames)
        
        write_ground_truth(os.path.join(output_dir, gt), downsampled_ground_truth)


if __name__ == "__main__":
    downsample_gt_cataract101()