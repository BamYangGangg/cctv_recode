#ffmpeng
import subprocess
import datetime
import requests
import json
import ffmpeg
from multiprocessing import Process
import torch

def record_rtsp_to_mp4(rtsp_url, output_file, duration):
    command = [
        'ffmpeg',
        '-rtsp_transport', 'tcp',
        '-i', rtsp_url,
        '-t', str(duration),
        '-c:v', 'copy',
        output_file
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("FFmpeg command failed: ", e)

def record_parallel(data, duration):
    currentTime = datetime.datetime.now()
    fileName = str(currentTime.strftime('%Y-%m-%d %H-%M-%S '))
    output_file = fileName + data[1] + '.mp4'
    record_rtsp_to_mp4(data[2], output_file, duration)

if __name__ == '__main__': 

    url = 'https://api.odcloud.kr/api/15063717/v1/uddi:fd7b941f-734e-4c1d-9155-975be33fc19c?page=1&perPage=8&serviceKey=ZhK1BEx2068JOjJ3DIakwiKM%2FIYNg%2Bx7JgqsVEb3gM7QkKcIHPICLSpnaEtZJgRTWjU4oSjNDC1ENv6YfW1e8A%3D%3D'
    response = requests.get(url)
    contents = response.text
    json_ob = json.loads(contents)
    data_list = [(i['CCTV관리번호'], i['설치위치주소'], i['스트리밍 프로토콜(RTSP)주소']) for i in json_ob['data']]
    duration = 3600
    total_duration = 24 * 3600

    start_time = datetime.datetime.now()

    processes = []
    while(datetime.datetime.now() - start_time).total_seconds() < total_duration:
        processes = []
        for data in data_list:
            process = Process(target=record_parallel, args= (data, duration))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()