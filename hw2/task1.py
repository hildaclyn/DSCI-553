import time
import sys
from pyspark import SparkContext
import os
from itertools import combinations
from collections import Counter


def modified_apriori_output(all_itemsets):
    # Modify the output to include a count of 1 for each itemset
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
    candidate_s= [set(candidate) for candidate in candidates]
    itemset = {candidate: 0 for candidate in candidates}
    baskets = [set(basket) for basket in baskets]
    for basket in baskets:
        for candidate, candidate_set in zip(candidates, candidate_s):
            if candidate_set.issubset(basket):
                itemset[candidate] += 1
    itemset = list(itemset.items())
    return itemset


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("There is an error for the Usage: python3 task1.py <case_number> <support> <input_filepath> <output_filepath>")
        sys.exit(1)

    sc = SparkContext(appName="Task1")
    time_start = time.time()
    
    case_num = sys.argv[1]
    support = sys.argv[2]
    input_filepath = sys.argv[3]
    output_filepath = sys.argv[4]

    try:
        case_num = int(case_num)
        support = int(support)
    except (ValueError, TypeError) as e:
        print(f"One or both strings are not integers: {e}")
    sc.setLogLevel("ERROR")
    csv_rdd = sc.textFile(input_filepath)

    header = csv_rdd.first() 
    csv_wt_head = csv_rdd.filter(lambda line: line != header)
    user_business = csv_wt_head.map(lambda line: line.split(","))

    data_list = None
    if case_num == 1:
        data_list = user_business.map(lambda x: (x[0], [x[1]])).reduceByKey(lambda a, b: a + b)
    elif case_num == 2:
        data_list = user_business.map(lambda x: (x[1], [x[0]])).reduceByKey(lambda a, b: a + b)
    data_l = data_list.map(lambda x: x[1])
   
    num_par = data_l.getNumPartitions()
    tot_bkt = data_l.count()

    candidates = data_l.mapPartitions(lambda partition: apriori(partition, tot_bkt, support)).reduceByKey(lambda x, y: x + y)
    candidates = candidates.map(lambda x: (x[0])).collect()
    candidates = sorted(candidates, key = lambda x: (len(x),) + tuple(str(x)))
    
    current = 1
    result = 'Candidates:\n'
    for i, candidate in enumerate(candidates):
        if len(candidate) != current:
            if i != 0:  # Avoid adding a newline
                result = result.rstrip(', ') + '\n\n' 
            current = len(candidate) 

        form_candidate = "('" + "', '".join(map(str, candidate)) + "')"
        result += form_candidate + ","

    result = result.rstrip(',') + '\n\n'

    Freq = data_l.mapPartitions(lambda partition: SON(partition, candidates)).reduceByKey(lambda x, y: x + y).filter(lambda x: x[1] >= support)
    #print(Freq.collect())
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
