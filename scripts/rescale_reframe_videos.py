import click
import subprocess
import os
from pathlib import Path

@click.command()
@click.argument('input_folder', type=click.Path(exists=True, file_okay=False))
@click.argument('output_folder', type=click.Path(file_okay=False))
@click.option('--fps', default=30, type=int, help='The target frame rate (e.g., 24, 30, 60).')
@click.option('--scale', default=1.0, type=float, help='The scaling factor (e.g., 0.5 for half size, 2.0 for double).')
def process_videos(input_folder, output_folder, fps, scale):
    """
    Processes all videos in INPUT_FOLDER, changes their framerate and scale, 
    and saves them to OUTPUT_FOLDER.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Define supported video extensions
    valid_extensions = {'.mp4', '.mkv', '.mov', '.avi', '.flv', '.wmv'}
    
    # Filter files in the directory
    files = [f for f in input_path.iterdir() if f.suffix.lower() in valid_extensions]

    if not files:
        click.echo("No valid video files found in the input folder.")
        return

    click.echo(f"Found {len(files)} videos. Starting processing...")

    with click.progressbar(files, label="Processing videos") as bar:
        for video_file in bar:
            output_file = output_path / video_file.name
            
            # Construct the FFmpeg command
            # 'trunc(iw*scale/2)*2' ensures the width/height is divisible by 2, 
            # which is required by many codecs like H.264.
            filter_chain = f"fps={fps},scale='trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2':flags=lanczos"
            
            cmd = [
                'ffmpeg',
                '-y',                 # Overwrite output files without asking
                '-i', str(video_file), # Input file path
                '-vf', filter_chain,   # Video filter (fps and scale)
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '18',
                '-stats -v', 'repeat+level+warning',
                str(output_file)       # Output file path
            ]

            try:
                # Run the command and suppress output unless there is an error
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                click.echo(f"\nError processing {video_file.name}: {e.stderr.decode()}", err=True)

    click.echo("\nAll tasks completed!")

if __name__ == '__main__':
    process_videos()
