import yaml
from src import util

def extract_wav_from_video(in_video_path, out_wav_path):
    util.cmd([
        'ffmpeg',
        '-i', in_video_path,
        '-vn',
        '-ac', 2,
        out_wav_path])

def get_video_stats(video_path):
    # TODO(3xg): This is brittle.
    stats = yaml.safe_load(util.system_result([
        'ffprobe',
        '-v','error',
        '-select_streams','v:0',
        '-show_entries','stream=width,height,r_frame_rate',
        '-of','json', 
        video_path]))['streams'][0]
    return dict(
        width=int(stats['width']),
        height=int(stats['height']),
        fps=int(stats['r_frame_rate'].split('/')[0]))

def images_to_video(image_folder, fps, video_path):
    util.cmd([
        'ffmpeg',
        '-framerate', fps,
        '-pattern_type', 'glob',
        '-i', image_folder + '/*.png',
        '-pix_fmt', 'yuv420p',
        '-v', 'info',
        '-y',
        video_path])

def mux_audio_video(in_audio_path, in_video_path, out_video_path):
    util.cmd([
        'ffmpeg',
        '-i', in_audio_path,
        '-i', in_video_path,
        '-c', 'copy',
        '-map', '0:a:0',
        '-map', '1:v:0',
        '-shortest', out_video_path])

def rescale_video(in_path, out_path, size):
    util.cmd([
        'ffmpeg',
        '-i', in_path,
        '-vf', 'scale={w}:{h},setsar=1:1'.format(w=size[0], h=size[1]),
        out_path])

def video_to_images(video_path, fps, image_folder):
    util.mkdirp(image_folder)
    util.cmd([
        'ffmpeg',
        '-i', video_path,
        '-vf', 'fps={f}'.format(f=fps),
        '{i}/%08d.png'.format(i=image_folder)])

