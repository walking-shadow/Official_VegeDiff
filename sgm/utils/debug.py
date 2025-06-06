import shutil
import kornia
from videoio_util import *
from ceph_util import PetrelBackend

loadp=PetrelBackend()
reader=PetrelVideoReader(file_client=loadp, num_threads=1)
x=reader.sample_whole_video(filename="s3://infdata/video/webvid10m0/00001/00001996.mp4")
y=reader.sample_clip(clip_length=10, clip_FPS_reate=2, filename="s3://infdata/video/webvid10m0/00001/00001996.mp4")

frames_tensor = np.asarray_chkfinite(x['imgs'], dtype=np.uint8)
frames_tensor = kornia.image_to_tensor(frames_tensor, keepdim=False).div(255.0)
print(frames_tensor)


# print(x['avg_fps'])
# import os
# import imageio
# import numpy as np

# def convert_frames_to_video(frames, output_path):
#     # 创建一个临时文件夹以保存每一帧的图像文件
#     temp_folder = "./temp_frames/"
#     os.makedirs(temp_folder, exist_ok=True)
#     pics_list = []
#     # 将每一帧保存为临时图像文件
#     for i, frame in enumerate(frames):
#         imageio.imwrite(f"{temp_folder}{i}.png", frame)
#         # pics_list.append(frame)

#     # 通过读取临时文件夹中的图像文件创建视频
#     with imageio.get_writer(output_path, mode='I', fps=30) as writer:
#         for i in range(len(frames)):
#             image = imageio.imread(f"{temp_folder}{i}.png")
#             pics_list.append(image)
#             writer.append_data(image)

#     # imageio.mimsave(output_path, pics_list, duration=1000/30, loop=0)  
#     # # 删除临时文件夹和其中的图像文件
    
#     shutil.rmtree(temp_folder)

# # 假设frames是包含n帧图像的列表，每帧都是一个numpy数组
# # frames = [np.array([0, 0, 0]), np.array([255, 255, 255]), np.array([128, 128, 128])]  # 这里只是一个示例

# # 将frames转换为MP4文件
# convert_frames_to_video(x["imgs"], "a.mp4")

