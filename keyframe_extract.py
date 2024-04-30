import subprocess

ffmpeg_cmd = [
    'ffmpeg',
    '-i', './2024-04-06-13-19-46_충청남도_천안시_동남구_신부동_433-4.mp4',
    '-vf', 'select=eq(pict_type\\,I)',
    '-vsync', 'vfr',
    'keyf_%03d.jpg'
]

subprocess.run(ffmpeg_cmd)