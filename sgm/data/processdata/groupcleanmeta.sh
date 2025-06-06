#!/bin/bash

# 指定输入文件的路径
input_file="/mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean_meta.txt"

# 指定输出文件的前缀
output_prefix="/mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean_meta.txt"

# 计算每个输出文件应包含的行数
total_lines=$(wc -l < "$input_file")
lines_per_file=$((total_lines / 16 + 1))

# 使用split命令将文件分割成8个部分
split --lines="$lines_per_file" "$input_file" "$output_prefix"

echo "文件分割完成。"