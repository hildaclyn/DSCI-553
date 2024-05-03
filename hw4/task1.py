import sys
from pyspark import SparkContext
import os
import time
from itertools import combinations
from pyspark.sql import SparkSession
from graphframes import GraphFrame


if __name__ == "__main__":

    #check the validation of the command
    if len(sys.argv) != 4:
        print("There is an error for the Usage: python3 task1.py <threshold> <input_filepath> <output_filepath>")
        sys.exit(1)

    sc = SparkContext(appName="Task1")

    thr = int(sys.argv[1])
    inputs = sys.argv[2]
    output = sys.argv[3]

    start_time = time.time()
    sc.setLogLevel("ERROR")
    csv_rdd = sc.textFile(inputs)
    header = csv_rdd.first()
    data_nh = (csv_rdd.filter(lambda x: x != header).map(lambda x: x.split(","))
               .map(lambda x: (x[0], {x[1]}))
               .reduceByKey(lambda a, b: a.union(b)))

    data_pre = data_nh.filter(lambda x: len(x[1]) >= thr).collectAsMap()

    uni_user = list(data_pre.keys())
    combuser = list(combinations(uni_user, 2))

    filter_p = sc.parallelize(combuser).filter(lambda users: len(data_pre[users[0]].intersection(data_pre[users[1]])) >= thr)
    edges_n = filter_p.flatMap(lambda users: [(users[1], users[0]), (users[0], users[1])])
    nodes = filter_p.flatMap(lambda users: [users[0], users[1]]).distinct().map(lambda user: (user,))

    spark = SparkSession.builder.appName("community").getOrCreate()
    vertices = spark.createDataFrame(nodes, ["id"])
    edges = spark.createDataFrame(edges_n, ["src", "dst"])
    g = GraphFrame(vertices, edges)
    result = g.labelPropagation(maxIter=5)
    final = result.rdd.map(lambda x: (x['label'], [x['id']]))
    final = final.reduceByKey(lambda a, b: a + b).map(lambda x: (sorted(x[1]), len(x[1]))).sortBy(lambda x: (x[1], x[0][0]))
    fin = final.map(lambda x: x[0])

    with open(output, 'w') as file:
        for f in fin.collect():
            com = ', '.join([f"'{u}'" for u in f])
            file.write(com + "\n")
    spark.stop()

    end_time = time.time()
    execution = end_time - start_time
    print(f"Duration: {execution}")
    sc.stop()
