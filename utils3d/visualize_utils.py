
from tqdm import tqdm
import os
import cv2
from moviepy.editor import VideoFileClip
from tqdm import tqdm

def visualize_image_sequence(image_seq_path, size=(512, 512), fps=30, save_dirs="./", process_func=None, convert_to_mp4=True, **args):
    # if is a dir, automatically create a video named output.avi
    # else save the video to the path
    # porcess_func: a function that process the image, args: image, n, **args
    os.makedirs(os.path.dirname(save_dirs), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = os.path.join(save_dirs, "output.avi") if os.path.isdir(save_dirs) else save_dirs
    video_out = None
    print("processing: ", output_path)
    for n, image_path in enumerate(tqdm(image_seq_path)):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if size is not None:
            image = cv2.resize(image, size)
        if process_func is not None:
            image = process_func(image, n, **args)
        if video_out is None:
            video_out = cv2.VideoWriter(output_path, fourcc, fps, (image.shape[1], image.shape[0]))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video_out.write(image)
    # use moviepy to convert the video to mp4
    video_out.release()
    if convert_to_mp4:
        clip = VideoFileClip(output_path)
        clip.write_videofile(output_path.replace(".avi", ".mp4"), codec="libx264", fps=fps)
        clip.close()
        # delete avi file
        os.remove(output_path)