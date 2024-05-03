import sys
from pyspark import SparkContext
import os
import time
import random
from itertools import combinations

def merge_dict(x, y):
    # copy of the first dictionary
    x.update(y)
    return x

def hash_f(user_index, a, b, p, m):
    return ((a * user_index + b) % p) % m

def band(id, hashvalues):
    bands = []
    for i in range(b):
        band_part = tuple(hashvalues[i * r : (i + 1) * r])
        bands.append(((i, band_part), [id]))
    return bands


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("There is an error for the Usage: python3 task1.py <input_filepath> <output_filepath>")
        sys.exit(1)
    sc = SparkContext(appName="Task1")
    input_filepath = sys.argv[1]
    output_filepath = sys.argv[2]

    start_time = time.time()
    sc.setLogLevel("ERROR")
    csv_rdd = sc.textFile(input_filepath)
    header = csv_rdd.first()
    no_head = csv_rdd.filter(lambda line: line != header)
    data_rdd = no_head.map(lambda line: (line.split(',')[1], {line.split(',')[0]}))
    bus_dict = data_rdd.reduceByKey(lambda a, b: merge_dict(a, b))

    uni_user = no_head.map(lambda line: line.split(',')[0]).distinct()
    user_ind = uni_user.zipWithUniqueId()
    hash_number = 50
    m = bus_dict.count()
    p = 1e5 + 3
    random.seed(42)
    hash_a_b = [(random.randint(1, m), random.randint(0, m)) for i in range(hash_number)]
    user_broadcast = sc.broadcast(dict(user_ind.collect()))
    
    hash_result = sc.parallelize([])

    for index, (a, b) in enumerate(hash_a_b):
        min_hash = bus_dict.flatMap(
            lambda x: [((x[0], index), int(hash_f(user_broadcast.value.get(user, -1), a, b, p, m)))for user in x[1]]).reduceByKey(
            min)
        
        if hash_result.isEmpty():
            hash_result = min_hash
        else:
            hash_result = hash_result.union(min_hash)
    gp_hash = hash_result.map(lambda x: (x[0][0], (x[0][1], x[1]))).groupByKey().mapValues(list)
    srt_hash = gp_hash.map(lambda x: (x[0], sorted(x[1], key=lambda y: y[0])))

    final = srt_hash.map(lambda x: (x[0], [y[1] for y in x[1]]))
    r = 2
    b = 25
    b_rdd = final.flatMap(lambda x: band(x[0], x[1]))
    
    add_r = b_rdd.reduceByKey(lambda a, b: a + b).filter(lambda x: len(x[1]) > 1)
    pairs = add_r.flatMap(lambda x: [tuple(sorted(pair)) for pair in combinations(x[1], 2)]).distinct()
    uni_pair = set(pairs.collect())
    bus_again = dict(bus_dict.collect())
    res = set()
    for pair in uni_pair:
        user1 = bus_again.get(pair[0])
        user2 = bus_again.get(pair[1])
        intersection = user1.intersection(user2)

        union = user1.union(user2)
        similarity = len(intersection) / len(union)
        if similarity >= 0.5:
            res.add((pair, similarity))

    sort_res = sorted(res, key=lambda x: (x[0][0], x[0][1]))

    na = "business_id_1,business_id_2,similarity\n"
    for pair in sort_res:
        business_id_1, business_id_2 = pair[0]
        similarity = pair[1]
        na += f"{business_id_1},{business_id_2},{similarity}\n"

    with open(output_filepath, 'w') as file:
        file.write(na)

    end_time = time.time()
    execution = end_time - start_time
    print(f"Duration: {execution}")
    sc.stop()
