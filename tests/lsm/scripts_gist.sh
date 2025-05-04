# data conversion
./fvecs_to_bin /data/hlqiu/diskann/gist/gist/gist_base.fvecs /data/hlqiu/diskann/gist/gist/gist_base.bin
./fvecs_to_bin /data/hlqiu/diskann/gist/gist/gist_query.fvecs /data/hlqiu/diskann/gist/gist/gist_query.bin
./ivecs_to_bin /data/hlqiu/diskann/gist/gist/gist_groundtruth.ivecs /data/hlqiu/diskann/gist/gist/gist_groundtruth.bin

./compute_groundtruth float /home/hlqiu/data/gist/gist_base.bin /home/hlqiu/data/gist/gist_query.bin 100000 /data/hlqiu/diskann/gist/index/gist_groundtruth.bin

# merge insert test
./partition_bin_dataset float /data/hlqiu/diskann/gist/gist/gist_base.bin 100k
## diskann
./build_disk_index float /data/hlqiu/diskann/gist/gist/gist_base_100k.bin ~/index/diskann_merge_insert_gist_test/diskann 63 75 100 500 64 l2 0 /data/hlqiu/diskann/gist/gist/gist_base_100k.tags

timestamp=$(date +"%Y%m%d%H%M")
nohup ./test_concurr_merge_insert float ~/index/tmp/ ~/index/diskann_merge_insert_gist_test/diskann ~/index/diskann_merge_insert_gist_test/diskann_merged ~/index/diskann_merge_insert_gist_test/mem_index /home/hlqiu/LSMIndex/log/${timestamp}_diskann_merge_insert_gist_test 75 1.2 75 1.2 ~/data/gist/gist_base.bin 0 ~/data/gist/gist_query.bin /data/hlqiu/diskann/gist/index/gist_groundtruth.bin 100 7168 0 63 5 2 75 2>&1 0</dev/null 1>~/LSMIndex/log/${timestamp}_diskann_merge_insert_gist_test.log &

## lsm_index
./build_disk_index float /data/hlqiu/diskann/gist/gist/gist_base_100k.bin ~/index/lsmidx_merge_insert_gist_test/level1_0 63 75 100 500 64 l2 0 /data/hlqiu/diskann/gist/gist/gist_base_100k.tags

timestamp=$(date +"%Y%m%d%H%M")
nohup ./test_concurr_merge_insert_delete_search_lsm_index float /home/hlqiu/index lsmidx_merge_insert_gist_test /home/hlqiu/LSMIndex/log/${timestamp}_lsmidx_merge_insert_gist_test ~/data/gist/gist_base.bin ~/data/gist/gist_query.bin /data/hlqiu/diskann/gist/index/gist_groundtruth.bin 75 1.2 75 1.2 0 100 7168 0 63 5 0 75 2>&1 0</dev/null 1>~/LSMIndex/log/${timestamp}_lsmidx_merge_insert_gist_test.log &

# merge insert delete test

# diskann
timestamp=$(date +"%Y%m%d%H%M")
nohup ./test_concurr_merge_insert float ~/index/tmp/ ~/index/diskann_merge_insert_gist_test/diskann ~/index/diskann_merge_insert_gist_test/diskann_merged ~/index/diskann_merge_insert_gist_test/mem_index /home/hlqiu/LSMIndex/log/${timestamp}_diskann_merge_insert_delete_gist_test 75 1.2 75 1.2 ~/data/gist/gist_base.bin 0 ~/data/gist/gist_query.bin /data/hlqiu/diskann/gist/index/gist_groundtruth.bin 100 3072 3072 63 5 3 75 2>&1 0</dev/null 1>~/LSMIndex/log/${timestamp}_diskann_merge_insert_delete_gist_test.log &

# lsm index
nohup ./test_concurr_merge_insert_delete_search_lsm_index float /home/hlqiu/index lsmidx_merge_insert_gist_test /home/hlqiu/LSMIndex/log/${timestamp}_lsmidx_merge_insert_delete_gist_test ~/data/gist/gist_base.bin ~/data/gist/gist_query.bin /data/hlqiu/diskann/gist/index/gist_groundtruth.bin 75 1.2 75 1.2 0 100 3072 3072 63 5 1 75 2>&1 0</dev/null 1>~/LSMIndex/log/${timestamp}_lsmidx_merge_insert_delete_gist_test.log &

# 消融
nohup ./test_concurr_merge_insert_delete_search_lsm_index float /home/hlqiu/index lsmidx_merge_insert_gist_test /home/hlqiu/LSMIndex/log/${timestamp}_lsmidx_merge_insert_delete_disable_relayout_gist_test ~/data/gist/gist_base.bin ~/data/gist/gist_query.bin /data/hlqiu/diskann/gist/index/gist_groundtruth.bin 75 1.2 75 1.2 0 100 3072 3072 63 5 1 75 2>&1 0</dev/null 1>~/LSMIndex/log/${timestamp}_lsmidx_merge_insert_delete_disable_relayout_gist_test.log &