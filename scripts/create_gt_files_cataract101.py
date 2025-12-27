import click
import pandas as pd
import os
from os.path import join


@click.command()
@click.option('--mappings', help='File mapping phase names to indexes')
@click.option('--output_path', help='Path to output folder')
@click.option('--annotations_file', help='Path to original annotations')
@click.option('--video_info_file', help='Path to annotations for video info')
def run(mappings, output_path, annotations_file, video_info_file):
    mappings = load_mappings(mappings)
    df_annos = pd.read_csv(annotations_file, sep=";")
    vidinfo_df = pd.read_csv(video_info_file, sep=";")

    cur_video = -1
    last_index = 0
    last_phase = -1
    cur_anno = []

    for _, row in df_annos.iterrows():
        video = row['VideoID']
        frame = row['FrameNo']
        phase = row['Phase']

        if video != cur_video: # new video found
            if cur_anno: # if this was not the first line
                vid_len = get_video_length(vidinfo_df, cur_video, last_index)
                cur_anno.extend([mappings[12]] * (vid_len - last_index + 1))
                write_annotations(output_path, cur_video, cur_anno)
                cur_anno = []
                last_index = 0
            cur_anno.extend([mappings[11]] * (frame - 1))
            cur_video = video
        else: # same video
            cur_anno.extend([mappings[last_phase]] * (frame - last_index))
        last_index = frame
        last_phase = phase

    vid_len = get_video_length(vidinfo_df, cur_video, last_index)
    cur_anno.extend([mappings[12]] * (vid_len - last_index + 1))
    write_annotations(output_path, cur_video, cur_anno)

def load_mappings(mappings_file):
    with open(mappings_file, "r") as file:
        raw_map = file.read().split("\n")
    return {int(item.split(" ")[0]): item.split(" ")[1] for item in raw_map if item != ""}

def get_video_length(vidinfo_df, video_id, default_length):
    vid_len = vidinfo_df.query(f"VideoID == {video_id}")['Frames']
    return vid_len.values[0] if len(vid_len) > 0 else default_length

def write_annotations(output_path, video_id, annotations):
    file_name = f"case_{video_id}.test.txt"
    with open(join(output_path, file_name), "w") as file:
        file.write("\n".join(annotations))

if __name__ == '__main__':
    run()
