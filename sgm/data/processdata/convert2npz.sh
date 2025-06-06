PARTITION=$1
QUOTA_TYPE=$2
JOB_NAME=$3
CPUS_PER_TASK=32
T=`date +%m%d%H%M`
PROJECT_NAME="${JOB_NAME}_${T}"

metadir=$4

mkdir logs/convert2npz/${JOB_NAME}
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=${QUOTA_TYPE} \
    --async \
    --output="logs/convert2npz/${JOB_NAME}/${PROJECT_NAME}.out" \
    --error="logs/convert2npz/${JOB_NAME}/${PROJECT_NAME}.err" \
    -x SH-IDC1-10-140-24-18 \
    python sgm/data/processdata/convert2npz.py \
    --project_name ${JOB_NAME} --metadir ${metadir}
# sh sgm/data/processdata/convert2npz.sh ai4earth reserved 0-convert2npz /mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean0_meta.txt
# sh sgm/data/processdata/convert2npz.sh ai4earth reserved 1-convert2npz /mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean1_meta.txt
# sh sgm/data/processdata/convert2npz.sh ai4earth reserved 2-convert2npz /mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean2_meta.txt
# sh sgm/data/processdata/convert2npz.sh ai4earth reserved 3-convert2npz /mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean3_meta.txt
# sh sgm/data/processdata/convert2npz.sh ai4earth reserved 4-convert2npz /mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean4_meta.txt
# sh sgm/data/processdata/convert2npz.sh ai4earth reserved 5-convert2npz /mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean5_meta.txt
# sh sgm/data/processdata/convert2npz.sh ai4earth reserved 6-convert2npz /mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean6_meta.txt
# sh sgm/data/processdata/convert2npz.sh ai4earth reserved 7-convert2npz /mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean7_meta.txt
# sh sgm/data/processdata/convert2npz.sh ai4earth reserved 8-convert2npz /mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean8_meta.txt
# sh sgm/data/processdata/convert2npz.sh ai4earth reserved 9-convert2npz /mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean9_meta.txt
# sh sgm/data/processdata/convert2npz.sh ai4earth reserved 10-convert2npz /mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean10_meta.txt
# sh sgm/data/processdata/convert2npz.sh ai4earth reserved 11-convert2npz /mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean11_meta.txt
# sh sgm/data/processdata/convert2npz.sh ai4earth reserved 12-convert2npz /mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean12_meta.txt
# sh sgm/data/processdata/convert2npz.sh ai4earth reserved 13-convert2npz /mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean13_meta.txt
# sh sgm/data/processdata/convert2npz.sh ai4earth reserved 14-convert2npz /mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean14_meta.txt
# sh sgm/data/processdata/convert2npz.sh ai4earth reserved 15-convert2npz /mnt/petrelfs/luzeyu/workspace/generative-models/dataset/meta/webvid10m_clean15_meta.txt