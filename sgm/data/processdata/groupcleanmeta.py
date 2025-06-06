import os
from tqdm import tqdm

with open("/mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean_meta.txt", "r") as file:
    clean_path_list = file.readlines()
clean_path_list = [line.strip() for line in clean_path_list]
print("clean:", len(clean_path_list))
num=len(clean_path_list)//8

clean_path_list0=clean_path_list[0:num]
clean_path_list1=clean_path_list[num:2*num]
clean_path_list2=clean_path_list[2*num:3*num]
clean_path_list3=clean_path_list[3*num:4*num]
clean_path_list4=clean_path_list[4*num:5*num]
clean_path_list5=clean_path_list[5*num:6*num]
clean_path_list6=clean_path_list[6*num:7*num]
clean_path_list7=clean_path_list[7*num:]

assert len(clean_path_list0)
+ len(clean_path_list1)
+ len(clean_path_list2)
+ len(clean_path_list3)
+ len(clean_path_list4)
+ len(clean_path_list5)
+ len(clean_path_list6)
+ len(clean_path_list7) == len(clean_path_list)

def write_list_to_file(lst, filename):
    with open(filename, 'w') as file:
        for item in tqdm(lst):
            file.write(str(item) + '\n')

print("0 start")
write_list_to_file(
    clean_path_list0,
    "/mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean0_meta.txt"
)

print("1 start")
write_list_to_file(
    clean_path_list1,
    "/mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean1_meta.txt"
)

print("2 start")
write_list_to_file(
    clean_path_list2,
    "/mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean2_meta.txt"
)

print("3 start")
write_list_to_file(
    clean_path_list3,
    "/mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean3_meta.txt"
)

print("4 start")
write_list_to_file(
    clean_path_list4,
    "/mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean4_meta.txt"
)

print("5 start")
write_list_to_file(
    clean_path_list5,
    "/mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean5_meta.txt"
)

print("6 start")
write_list_to_file(
    clean_path_list6,
    "/mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean6_meta.txt"
)

print("7 start")
write_list_to_file(
    clean_path_list7,
    "/mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean7_meta.txt"
)