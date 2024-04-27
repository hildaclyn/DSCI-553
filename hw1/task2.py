import json
import sys
from pyspark import SparkContext
import os
import time

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("There is an error for the Usage: python3 task2.py <review_filepath> <output_filepath> <n_partition>")
        sys.exit(1)

    sc = SparkContext(appName="Task2")

    review_filepath = sys.argv[1]
    output_filepath = sys.argv[2]
    n_partition_in = sys.argv[3]

    n_partition = int(n_partition_in)
    review_rdd = sc.textFile(review_filepath)

    start_def = time.time()
    buss_count_1 = review_rdd.map(lambda review: [json.loads(review)['business_id'], 1]).reduceByKey(
        lambda a, b: a + b)
    top10b = sc.parallelize(buss_count_1.collect())
    n_p = top10b.getNumPartitions()
    list_partition = top10b.glom()
    num_items = list_partition.map(lambda x: len(x))
    n_items = num_items.collect()
    top_10_def = top10b.sortBy(lambda x: (-x[1], x[0])).take(10)

    end_def = time.time()
    exe_time_def = end_def-start_def
    default = {
        "n_partition" : n_p,
        "n_items" : n_items,
        "exe_time" : exe_time_def
    }
#-----------------------------------------------------------------------------------------
    start_cus = time.time()

    buss_count_2 = review_rdd.map(lambda review: (json.loads(review)['business_id'], 1)) \
        .reduceByKey(lambda a, b: a + b)
    partitioned_rdd = buss_count_2.partitionBy(n_partition, lambda k: hash(k) % n_partition)
    n_items_cus = partitioned_rdd.glom().map(len).collect()
    top_10_cus = partitioned_rdd.sortBy(lambda x: (-x[1], x[0])).take(10)

    end_cus = time.time()
    exe_time_cus = end_cus - start_cus

    customized = {
        "n_partition": n_partition,
        "n_items": n_items_cus,
        "exe_time": exe_time_cus
    }
    comb_partition = {"default":default, "customized":customized}

]    with open(output_filepath, 'w', encoding='utf-8') as outputs:
        json.dump(comb_partition, outputs)

    sc.stop()
