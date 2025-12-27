import subprocess
from pathlib import Path
import click


@click.command()
@click.option("--input-dir", "-i", type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help="Folder containing the source videos.")
@click.option("--output-dir", "-o", type=click.Path(file_okay=False, dir_okay=True), required=True, help="Folder where converted videos will be saved.")
@click.option("--fps", "-f", default=15, show_default=True, help="Target framerate for output videos.")
def convert_videos(input_dir, output_dir, fps):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    video_exts = {".mp4", ".mov", ".avi", ".mkv"}

    for file in input_dir.iterdir():
        if file.suffix.lower() in video_exts:
            output_file = output_dir / f"{file.stem}.mp4"

            cmd = [
                "ffmpeg",
                "-y",
                "-i", str(file),
                "-vf", f"fps={fps}",        # deterministic decimation
                "-c:v", "libx264",
                "-crf", "18",
                #"-preset", "slow",
                #"-c:a", "copy",
                str(output_file),
            ]

            click.echo(f"Converting: {file.name} → {output_file.name}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print("FFmpeg command failed:")
                print(" ".join(cmd))
                print("---- FFmpeg stderr ----")
                print(e.stderr if hasattr(e, 'stderr') else "No stderr captured.")
                raise

    click.echo("All conversions completed!")


if __name__ == "__main__":
    convert_videos()

