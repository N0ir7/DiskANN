# data conversion
./bvecs_to_bin /data/hlqiu/diskann/sift1b_data/bigann_base.bvecs /data/hlqiu/diskann/sift1b_data/bigann_base.bin
./bvecs_to_bin /data/hlqiu/diskann/sift1b_data/bigann_query.bvecs /data/hlqiu/diskann/sift1b_data/bigann_query.bin
./ivecs_to_bin /data/hlqiu/diskann/sift1b_data/gnd/idx_100M.ivecs /data/hlqiu/diskann/sift1b_data/bigann_groundtruth.bin

./compute_groundtruth float /home/hlqiu/data/deep10m/deep10m_base.bin /home/hlqiu/data/deep10m/deep10m_query.bin 10000 /data/hlqiu/diskann/deep10m/deep10m_groundtruth.bin

# merge insert test
./partition_bin_dataset float /data/hlqiu/diskann/deep10m/deep10m_base.bin 1m
## diskann
./build_disk_index float /data/hlqiu/diskann/deep10m/deep10m_base_1m.bin ~/index/diskann_merge_insert_deep10m_test/diskann 63 75 100 500 64 l2 0 /data/hlqiu/diskann/deep10m/deep10m_base_1m.tags

timestamp=$(date +"%Y%m%d%H%M")
nohup ./test_concurr_merge_insert float ~/index/tmp/ ~/index/diskann_merge_insert_deep10m_test/diskann ~/index/diskann_merge_insert_deep10m_test/diskann_merged ~/index/diskann_merge_insert_deep10m_test/mem_index /home/hlqiu/LSMIndex/log/${timestamp}_diskann_merge_insert_deep10m_test 75 1.2 75 1.2 ~/data/deep10m/deep10m_base.bin 0 ~/data/deep10m/deep10m_query.bin /data/hlqiu/diskann/deep10m/deep10m_groundtruth.bin 100 71680 0 63 5 2 75 2>&1 0</dev/null 1>~/LSMIndex/log/${timestamp}_diskann_merge_insert_deep10m_test.log &

## lsm_index
./build_disk_index float /data/hlqiu/diskann/deep10m/deep10m_base_100k.bin ~/index/lsmidx_merge_insert_deep10m_test/level1_0 63 75 100 500 64 l2 0 /data/hlqiu/diskann/deep10m/deep10m_base_100k.tags

timestamp=$(date +"%Y%m%d%H%M")
nohup ./test_concurr_merge_insert_delete_search_lsm_index float /home/hlqiu/index lsmidx_merge_insert_deep10m_test /home/hlqiu/LSMIndex/log/${timestamp}_lsmidx_merge_insert_deep10m_test ~/data/deep10m/deep10m_base.bin ~/data/deep10m/deep10m_query.bin /data/hlqiu/diskann/deep10m/deep10m_groundtruth.bin 75 1.2 75 1.2 0 100 71680 0 63 5 0 75 2>&1 0</dev/null 1>~/LSMIndex/log/${timestamp}_lsmidx_merge_insert_deep10m_test.log &

# merge insert delete test

# diskann
timestamp=$(date +"%Y%m%d%H%M")
nohup ./test_concurr_merge_insert float ~/index/tmp/ ~/index/diskann_merge_insert_deep10m_test/diskann ~/index/diskann_merge_insert_deep10m_test/diskann_merged ~/index/diskann_merge_insert_deep10m_test/mem_index /home/hlqiu/LSMIndex/log/${timestamp}_diskann_merge_insert_delete_deep10m_test 75 1.2 75 1.2 ~/data/deep10m/deep10m_base.bin 0 ~/data/deep10m/deep10m_query.bin /data/hlqiu/diskann/deep10m/deep10m_groundtruth.bin 100 30720 30720 63 5 3 75 2>&1 0</dev/null 1>~/LSMIndex/log/${timestamp}_diskann_merge_insert_delete_deep10m_test.log &

# lsm index
nohup ./test_concurr_merge_insert_delete_search_lsm_index float /home/hlqiu/index lsmidx_merge_insert_deep10m_test /home/hlqiu/LSMIndex/log/${timestamp}_lsmidx_merge_insert_delete_deep10m_test ~/data/deep10m/deep10m_base.bin ~/data/deep10m/deep10m_query.bin /data/hlqiu/diskann/deep10m/deep10m_groundtruth.bin 75 1.2 75 1.2 0 100 30720 30720 63 5 1 75 2>&1 0</dev/null 1>~/LSMIndex/log/${timestamp}_lsmidx_merge_insert_delete_deep10m_test.log &

# 消融
nohup ./test_concurr_merge_insert_delete_search_lsm_index float /home/hlqiu/index lsmidx_merge_insert_deep10m_test /home/hlqiu/LSMIndex/log/${timestamp}_lsmidx_merge_insert_delete_deep10m_disable_redistribute_test ~/data/deep10m/deep10m_base.bin ~/data/deep10m/deep10m_query.bin /data/hlqiu/diskann/deep10m/deep10m_groundtruth.bin 75 1.2 75 1.2 0 100 30720 30720 63 5 1 75 2>&1 0</dev/null 1>~/LSMIndex/log/${timestamp}_lsmidx_merge_insert_delete_deep10m_disable_redistribute_test.log &
