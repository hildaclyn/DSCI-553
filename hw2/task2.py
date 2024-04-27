import time
import sys
from pyspark import SparkContext
import os
import csv
from itertools import combinations
from collections import Counter

def preprocess(lines):
    line = lines.split(',')
    date = line[0].strip('"').split('/')
    date = '/'.join([date[0], date[1], date[2][-2:]])
    Date_Customer_id = date + '-' + line[1].strip('"')
    Product_id = int(line[5].strip('"'))
    return (Date_Customer_id, Product_id)

def modified_apriori_output(all_itemsets):
    modified_output = [(itemset, 1) for itemset in all_itemsets]
    return modified_output


def apriori(baskets, tot_n, spt):
    baskets = [set(basket) for basket in baskets]
    num_p = len(baskets)
    sup = (num_p / tot_n) * spt

    item_dict = Counter(item for basket in baskets for item in basket)
    frequent_items = {item for item, count in item_dict.items() if count >= sup}
    all_itemsets = {(item,) for item in frequent_items}
    k = 2
    while True:
        candidate_mult = set()
        for itemset in all_itemsets:
            if len(itemset) == k - 1:
                for item in frequent_items:
                    new_candidate = tuple(sorted(set(itemset).union({item})))
                    if new_candidate not in candidate_mult and all(
                            tuple(sorted(comb)) in all_itemsets for comb in combinations(new_candidate, k - 1)):
                        candidate_mult.add(new_candidate)

        counts = Counter()
        for basket in baskets:
            for candidate in candidate_mult:
                if set(candidate).issubset(basket):
                    counts[candidate] += 1
        new_frequent = set(itemset for itemset, count in counts.items() if count >= sup)
        if not new_frequent:
            break

        all_itemsets |= set(new_frequent)
        k += 1

    return modified_apriori_output(all_itemsets)

def SON(baskets, candidates):
    candidate_s = [set(candidate) for candidate in candidates]
    itemset = {candidate: 0 for candidate in candidates}
    baskets = [set(basket) for basket in baskets]
    for basket in baskets:
        for candidate, candidate_set in zip(candidates, candidate_s):
            if candidate_set.issubset(basket):
                itemset[candidate] += 1
    itemset = list(itemset.items())
    return itemset

if __name__ == "__main__":
    #check the validation of the command
    if len(sys.argv) != 5:
        print("There is an error for the Usage: python3 task2.py <filter threshold> <support> <input_filepath> <output_filepath>")
        sys.exit(1)

    sc = SparkContext(appName="Task2")
    time_start = time.time()

    filter_thr = sys.argv[1]
    support = sys.argv[2]
    input_filepath = sys.argv[3]
    output_filepath = sys.argv[4]

    try:
        filter_thr = int(filter_thr)
        support = int(support)
    except (ValueError, TypeError) as e:
        print(f"One or both strings are not integers: {e}")
    csv_rdd = sc.textFile(input_filepath)
    header = csv_rdd.first() 
    csv_no_head = csv_rdd.filter(lambda line: line != header)

    filter_rdd = csv_no_head.map(preprocess)
    filter_csv = filter_rdd.collect()

    with open("Customer_with_Product.csv", 'w') as file:
        file.write('DATE-CUSTOMER_ID,PRODUCT_ID\n')
        for row in filter_csv:
            file.write(f'{row[0]},{row[1]}\n')

    cus_p = filter_rdd.map(lambda x: (x[0], [x[1]])).reduceByKey(lambda a, b: a + b).filter(lambda x: len(x[1]) > filter_thr)
    cus_prod = cus_p.map(lambda x: x[1])

    num_par = cus_prod.getNumPartitions()

    tot_bkt = cus_prod.count()

    candid = cus_prod.mapPartitions(lambda partition: apriori(partition, tot_bkt, support)).reduceByKey(
        lambda x, y: x + y)
    candida = candid.map(lambda x: (x[0])).collect()
    candidates = sorted(candida, key = lambda x: (len(x),) + tuple(str(x)))

    current = 1
    result = 'Candidates:\n'
    for i, candidate in enumerate(candidates):
        if len(candidate) != current:
            if i != 0:
                result = result.rstrip(', ') + '\n\n' 
            current = len(candidate) 

        #  the candidate as a string and append to the result
        form_candidate = "('" + "', '".join(map(str, candidate)) + "')"
        result += form_candidate + ","

    result = result.rstrip(',') + '\n\n'

    Freq = cus_prod.mapPartitions(lambda partition: SON(partition, candidates)).reduceByKey(lambda x, y: x + y).filter(
        lambda x: x[1] >= support)
    sort_freq = Freq.map(lambda x: (x[0])).collect()
    
    Freq_filter = sorted(sort_freq, key = lambda x: (len(x),) + tuple(str(x)))

    current_2 = 1
    result += 'Frequent Itemsets:\n'
    for i, fre in enumerate(Freq_filter):
        if len(fre) != current_2:
            if i != 0:
                result = result.rstrip(', ') + '\n\n' 
            current_2 = len(fre)

        form_fre = "('" + "', '".join(map(str, fre)) + "')"
        result += form_fre + ","
    result = result.rstrip(',')

    with open(output_filepath, 'w+') as f:
        f.write(result)

    time_end = time.time()
    print("Duration: ", time_end - time_start)

    sc.stop()
