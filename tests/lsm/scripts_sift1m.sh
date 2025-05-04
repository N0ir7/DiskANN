# data conversion
./fvecs_to_bin /home/hlqiu/data/sift/sift_query.fvecs /home/hlqiu/data/sift/sift_query.bin
./ivecs_to_bin /home/hlqiu/data/sift/sift_groundtruth.ivecs /home/hlqiu/data/sift/sift_groundtruth.bin

./compute_groundtruth float /home/hlqiu/data/sift/sift_base.bin /home/hlqiu/data/sift/sift_query.bin 100000 /data/hlqiu/diskann/data/sift/sift_groundtruth.bin

# dataset partition
./partition_bin_dataset float ~/data/sift_lsmidx_merge_test/sift_base.bin 32k 100k 32k 32k 32k 32k 
./partition_bin_dataset float ~/data/sift_lsmidx_level0_2_test/sift_base.bin 32k 32k 32k
./partition_bin_dataset float ~/data/sift_lsmidx_merge_test2/sift_base.bin 132k 32k 32k 32k 32k 

## build diskann
cp ~/index/lsm/mem_index ~/index/diskann/
cp ~/index/lsm/mem_index.data ~/index/diskann/
cp ~/index/lsm/mem_index.tags ~/index/diskann/
./build_disk_index float ~/data/sift/sift_base_968k.bin ~/index/diskann/diskann 64 75 100 500 64 l2 0 ~/data/sift/sift_base_968k.tags

# merge insert test
./partition_bin_dataset ~/data/sift_merge_insert_test/sift_base.bin 100k
## diskann
./build_disk_index float ~/data/sift_merge_insert_test/sift_base_100k.bin ~/index/diskann_merge_insert_test/diskann 64 75 100 500 64 l2 0 ~/data/sift_merge_insert_test/sift_base_100k.tags

timestamp=$(date +"%Y%m%d%H%M")
nohup ./test_concurr_merge_insert float ~/index/tmp/ ~/index/diskann_merge_insert_test/diskann ~/index/diskann_merge_insert_test/diskann_merged ~/index/diskann_merge_insert_test/mem_index /home/hlqiu/LSMIndex/log/${timestamp}_diskann_merge_insert_test 75 1.2 75 1.2 ~/data/sift/sift_base.bin 0 ~/data/sift/sift_query.bin /data/hlqiu/diskann/data/sift/sift_groundtruth.bin 100 7168 0 64 5 2 75 2>&1 0</dev/null 1>~/LSMIndex/log/${timestamp}_diskann_merge_insert_test.log &

## lsm_index
./build_disk_index float ~/data/sift_merge_insert_test/sift_base_100k.bin ~/index/lsmidx_merge_insert_test/level1_0 64 75 100 500 64 l2 0 ~/data/sift_merge_insert_test/sift_base_100k.tags

timestamp=$(date +"%Y%m%d%H%M")
nohup ./test_concurr_merge_insert_delete_search_lsm_index float /home/hlqiu/index lsmidx_merge_insert_test /home/hlqiu/LSMIndex/log/${timestamp}_lsmidx_merge_insert_test /home/hlqiu/data/sift/sift_base.bin /home/hlqiu/data/sift/sift_query.bin /home/hlqiu/data/sift/sift_groundtruth.bin 75 1.2 75 1.2 0 100 7168 0 64 5 0 75 2>&1 0</dev/null 1>~/LSMIndex/log/${timestamp}_lsmidx_merge_insert_test.log &

# merge insert delete test

# diskann
timestamp=$(date +"%Y%m%d%H%M")
nohup ./test_concurr_merge_insert float ~/index/tmp/ ~/index/diskann_merge_insert_delete_test/diskann ~/index/diskann_merge_insert_delete_test/diskann_merged ~/index/diskann_merge_insert_delete_test/mem_index /home/hlqiu/LSMIndex/log/${timestamp}_diskann_merge_insert_delete_test 75 1.2 75 1.2 ~/data/sift/sift_base.bin 0 ~/data/sift/sift_query.bin /data/hlqiu/diskann/data/sift/sift_groundtruth.bin 100 3072 3072 64 5 3 75 2>&1 0</dev/null 1>~/LSMIndex/log/${timestamp}_diskann_merge_insert_delete_test.log &

# lsm index
nohup ./test_concurr_merge_insert_delete_search_lsm_index float /home/hlqiu/index lsmidx_merge_insert_test /home/hlqiu/LSMIndex/log/${timestamp}_lsmidx_merge_insert_delete_test /home/hlqiu/data/sift/sift_base.bin /home/hlqiu/data/sift/sift_query.bin /home/hlqiu/data/sift/sift_groundtruth.bin 75 1.2 75 1.2 0 100 3072 3072 64 5 1 75 2>&1 0</dev/null 1>~/LSMIndex/log/${timestamp}_lsmidx_merge_insert_delete_test.log &

# 启发实验
./partition_bin_dataset float ~/data/sift/sift_base.bin 100k 500k
./build_disk_index float ~/data/sift/sift_base_500k.bin ~/index/lsm_search_500k/level1_0 64 75 100 500 64 l2 0 ~/data/sift/sift_base_500k.tags
./build_disk_index float ~/data/sift/sift_base_1m.bin ~/index/lsm_search_1m/level1_0 64 75 100 500 64 l2 0 ~/data/sift/sift_base_1m.tags
nohup ./test_search_lsm_index float ~/index lsmidx_merge_insert_test_backup ~/data/sift/sift_query.bin /data/hlqiu/diskann/data/sift/sift_groundtruth.bin /home/hlqiu/LSMIndex/log/${timestamp}_lsmidx_search_test 0 10 64 1.2 5 5 15 25 35 45 55 65 75 2>&1 0</dev/null 1>~/LSMIndex/log/${timestamp}_lsmidx_search_test.log &

nohup ./test_search_lsm_index float ~/index lsm_search_500k ~/data/sift/sift_query.bin /data/hlqiu/diskann/data/sift/sift_groundtruth.bin /home/hlqiu/LSMIndex/log/${timestamp}_lsmidx_500k_search_test 0 10 64 1.2 5 5 15 25 35 45 55 65 75 2>&1 0</dev/null 1>~/LSMIndex/log/${timestamp}_lsmidx_500k_search_test.log &

nohup ./test_search_lsm_index float ~/index lsm_search_1m ~/data/sift/sift_query.bin /data/hlqiu/diskann/data/sift/sift_groundtruth.bin /home/hlqiu/LSMIndex/log/${timestamp}_lsmidx_1m_search_test 0 10 64 1.2 5 5 15 25 35 45 55 65 75 2>&1 0</dev/null 1>~/LSMIndex/log/${timestamp}_lsmidx_1m_search_test.log &

# 消融测试
timestamp=$(date +"%Y%m%d%H%M")

# 动态搜索参数
nohup ./test_concurr_merge_insert_delete_search_lsm_index float /home/hlqiu/index lsmidx_merge_insert_test /home/hlqiu/LSMIndex/log/${timestamp}_lsmidx_merge_insert_disable_skip_test /home/hlqiu/data/sift/sift_base.bin /home/hlqiu/data/sift/sift_query.bin /home/hlqiu/data/sift/sift_groundtruth.bin 75 1.2 75 1.2 0 100 7168 0 64 5 0 75 2>&1 0</dev/null 1>~/LSMIndex/log/${timestamp}_lsmidx_merge_insert_disable_skip_test.log &

nohup ./test_concurr_merge_insert_delete_search_lsm_index float /home/hlqiu/index lsmidx_merge_insert_test /home/hlqiu/LSMIndex/log/${timestamp}_lsmidx_merge_insert_disable_dynamic_test /home/hlqiu/data/sift/sift_base.bin /home/hlqiu/data/sift/sift_query.bin /home/hlqiu/data/sift/sift_groundtruth.bin 75 1.2 75 1.2 0 100 7168 0 64 5 0 75 2>&1 0</dev/null 1>~/LSMIndex/log/${timestamp}_lsmidx_merge_insert_disable_dynamic_test.log &

# 重布局
nohup ./test_concurr_merge_insert_delete_search_lsm_index float /home/hlqiu/index lsmidx_merge_insert_test /home/hlqiu/LSMIndex/log/${timestamp}_lsmidx_merge_insert_delete_disable_redistribute_test /home/hlqiu/data/sift/sift_base.bin /home/hlqiu/data/sift/sift_query.bin /home/hlqiu/data/sift/sift_groundtruth.bin 75 1.2 75 1.2 0 100 3072 3072 64 5 1 75 2>&1 0</dev/null 1>~/LSMIndex/log/${timestamp}_lsmidx_merge_insert_delete_disable_redistribute_test.log &