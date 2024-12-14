# 转换数据格式
./fvecs_to_bin /home/hlqiu/data/sift/sift_query.fvecs /home/hlqiu/data/sift/sift_query.bin
./ivecs_to_bin /home/hlqiu/data/sift/sift_groundtruth.ivecs /home/hlqiu/data/sift/sift_groundtruth.bin

# 分割数据集
./partition_bin_dataset ~/data/sift/sift_base.bin 32k
./partition_bin_dataset ~/data/sift/sift_base.bin 32k 320k
./partition_bin_dataset ~/data/sift_4/sift_base.bin 16k 160k 320k

# 构建2级lsm_index
./build_disk_index float ~/data/sift/sift_base_320k.bin ~/index/lsm/level1 64 75 100 500 64 l2 0 ~/data/sift/sift_base_320k.tags
./build_disk_index float ~/data/sift/sift_base_648k.bin ~/index/lsm/level2 64 75 100 500 64 l2 0 ~/data/sift/sift_base_648k.tags
./build_memory_index float ~/data/sift/sift_base_32k.bin ~/data/sift/sift_base_32k.tags ~/index/lsm/mem_index_0 1 0 64 75 1.2 64 l2

# 构建3级lsm_index
./build_disk_index float ~/data/sift_4/sift_base_160k.bin ~/index/lsm2/level1 64 75 100 500 64 l2 0 ~/data/sift_4/sift_base_160k.tags

./build_disk_index float ~/data/sift_4/sift_base_320k.bin ~/index/lsm2/level2 64 75 100 500 64 l2 0 ~/data/sift_4/sift_base_320k.tags

./build_disk_index float ~/data/sift_4/sift_base_504k.bin ~/index/lsm2/level3 64 75 100 500 64 l2 0 ~/data/sift_4/sift_base_504k.tags

./build_memory_index float ~/data/sift_4/sift_base_16k.bin ~/data/sift_4/sift_base_16k.tags ~/index/lsm2/mem_index_0 1 0 64 75 1.2 64 l2

# 构建diskann
cp ~/index/lsm/mem_index ~/index/diskann/
cp ~/index/lsm/mem_index.data ~/index/diskann/
cp ~/index/lsm/mem_index.tags ~/index/diskann/
./build_disk_index float ~/data/sift/sift_base_968k.bin ~/index/diskann/diskann 64 75 100 500 64 l2 0 ~/data/sift/sift_base_968k.tags

# diskann执行搜索
mkdir -p ~/log
nohup ./test_concurr_merge_insert float ~/index/tmp ~/index/diskann/diskann ~/index/diskann/diskann_merged ~/index/diskann/mem_index 75 1.2 75 1.2 ~/data/sift/sift_base.bin 0 ~/data/sift/sift_query.bin ~/data/sift/sift_groundtruth.bin 10 0 0 64 5 1 75 2>&1 0</dev/null 1>~/log/$(date +"%Y%m%d%H%M")_diskann_search_test.log &

# 2级lsmidx执行搜索
nohup ./test_search_lsm_index float ~/index lsm ~/data/sift/sift_query.bin ~/data/sift/sift_groundtruth.bin 0 10 64 1.2 5 75 2>&1 0</dev/null 1>~/log/$(date +"%Y%m%d%H%M")_lsmidx_2level_search_test.log &

# 3级lsmidx执行搜索
nohup ./test_search_lsm_index float ~/index lsm2 ~/data/sift/sift_query.bin ~/data/sift/sift_groundtruth.bin 0 10 64 1.2 5 75 2>&1 0</dev/null 1>~/log/$(date +"%Y%m%d%H%M")_lsmidx_3level_search_test.log &

# 其他测试

./test /home/hlqiu/index/lsm/level1