import os
from tqdm import tqdm

with open("/mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_error_meta.txt", "r") as file:
    err_path_list = file.readlines()
err_path_list = [line.strip() for line in err_path_list]

with open("/mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_meta.txt", "r") as file:
    path_list = file.readlines()
path_list = [line.strip() for line in path_list]

print("err:", len(err_path_list))
print("normal:", len(path_list))

length_clean = 0
for path in tqdm(path_list):
    if path not in err_path_list:
        length_clean += 1
        with open("webvid10m_meta_clean.txt", "a+") as f:
            f.write(path+"\n")
    else:
        print(path)
print("length_clean", length_clean)