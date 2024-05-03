import copy
import sys
import os
import time
from pyspark import SparkContext
import random
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict

def cal_clst_stat(clusters):

    clst_s = {}
    clst_i = defaultdict(list)

    for label, features in clusters.items():
        features = np.array(features)
        if len(features) > 0:  # Ensure there are features to process
            SUM = np.sum(features, axis=0)
            SUMSQ = np.sum(features ** 2, axis=0)
            count_n = features.shape[0]
            centroids = SUM / count_n
            mean_sq = SUMSQ / count_n
            variance = mean_sq - np.square(centroids)

            # Populate cluster summary dictionary
            clst_s[label] = {
                "N": count_n,
                "SUM": SUM,
                "SUMSQ": SUMSQ,
                "centroids": centroids,
                "variance": variance
            }

            # Aggregate indices associated with each feature in the cluster
            for feature in features:
                fe = tuple(feature)
                clst_i[label] += fea_dict.get(fe, [])

    return clst_s, clst_i

def cal_len(ds_d, cs_d):
    sum_ds = sum(item["N"] for item in ds_d.values())
    cs_len = len(cs_d)
    sum_cs = sum(item["N"] for item in cs_d.values())
    return sum_ds, cs_len, sum_cs

def update_cluster_stats(cluster, point):
    new_cluster = cluster.copy()
    # Update cluster statistics with a new point """
    n = new_cluster["N"] + 1
    new_sum = new_cluster["SUM"] + point
    new_sumsq = new_cluster["SUMSQ"] + point**2

    new_cluster["N"] = n
    new_cluster["SUM"] = new_sum
    new_cluster["SUMSQ"] = new_sumsq
    new_cluster["centroids"] = new_sum / n
    new_cluster["variance"] = (new_sumsq / n) - np.square(new_sum / n)

    return new_cluster  # Make sure to return the updated cluster dictionary


def mahalanobis_dist(p, centr, vars):
    if np.any(vars == 0):
        raise ValueError("Variance values must be non-zero")
    squared_dist = np.sum(((p - centr) ** 2) / vars)
    return np.sqrt(squared_dist)

def cluster_cs(cluster1, cluster2):
    centroid_1 = cluster1['centroids']
    centroid_2 = cluster2['centroids']
    variance_1 = cluster1['variance']
    variance_2 = cluster2['variance']

    # Ensure variances are not zero to avoid division by zero
    variance_1 = np.where(variance_1 == 0, np.finfo(float).eps, variance_1)
    variance_2 = np.where(variance_2 == 0, np.finfo(float).eps, variance_2)

    diff = centroid_1 - centroid_2
    dist1 = np.sqrt(np.sum((diff ** 2) / variance_1))
    dist2 = np.sqrt(np.sum((diff ** 2) / variance_2))

    # Return the smaller of the two distances
    return min(dist1, dist2)


if __name__ == "__main__":

    sc = SparkContext(appName="Task1")
    inputs = sys.argv[1]
    n_cluster = int(sys.argv[2])
    outputs = sys.argv[3]

    sc.setLogLevel("ERROR")

    start_time = time.time()
    res_str = "The intermediate results:\n"
    data_rdd = sc.textFile(inputs)
    data_rdd = data_rdd.map(lambda line: line.split(','))

    # Convert data types
    # Assuming the first column is an integer ID and the rest are floating point numbers
    data_rdd = data_rdd.map(lambda x: (int(x[0]), tuple(map(float, x[2:]))))
    #{index:(features)}
    data_dict = data_rdd.collectAsMap()

    reversed_rdd = data_rdd.map(lambda x: (x[1], x[0]))
    rev = reversed_rdd.groupByKey().mapValues(list)
    fea_dict = rev.collectAsMap()
    #print(fea_dict)

    #array[features..]for each index
    feature_array = [np.array(features) for features in data_dict.values()]
    
#-------------------------------------------------------------------------------
    #print(feature_array)
    random.shuffle(feature_array)
    #print(feature_array)
    # Sample 20% of the data
    o_t_f = round(len(feature_array) * 0.2)
    first_d = feature_array[:o_t_f]

    km_f = KMeans(n_clusters=n_cluster * 5).fit(first_d)
    #print(km_f.labels_)
    labels = km_f.labels_
    clst_dict_1 = {}

    for label in labels:
        clst_dict_1[label] = clst_dict_1.get(label, 0) + 1
    #print(clst_dict_1)
    RS = []
    RI = []
    for index, label in enumerate(labels):
        if clst_dict_1[label] == 1:
            RS.append(first_d[index])
            RI.append(index)

            # Create a new list excluding the outliers
    new_d = np.array([point for idx, point in enumerate(first_d) if idx not in RI])

    # step 4
    km_2 = KMeans(n_clusters=n_cluster).fit(new_d)
    ds_lbs = km_2.labels_

    clst_pt_ds = {}
    for idx, label in enumerate(ds_lbs):
        clst_pt_ds[label] = clst_pt_ds.get(label, [])
        clst_pt_ds[label].append(new_d[idx])

    # Convert lists to numpy arrays for efficient computations
    for label in clst_pt_ds:
        clst_pt_ds[label] = clst_pt_ds[label]

    clst_sum_DS, clst_ind_DS = cal_clst_stat(clst_pt_ds)

    # step 6
    new_RS = []
    new_CS = defaultdict(list)
    clst_sum_CS = {}
    clst_ind_CS = defaultdict(list)
    kmRS_k = None
    l_k = n_cluster * 5  # 5 times the number of input clusters
    if len(RS) >= l_k:  # Check if there are enough points to perform K-Means
        kmRS_k = KMeans(n_clusters=l_k).fit(RS)
        # RS_lbs = kmRS_k.labels_
    elif len(RS) > 0:
            kmRS_k = KMeans(n_clusters=len(RS)).fit(RS)
    else:
        RS_lbs = []
    if kmRS_k is not None:
        RS_lbs = kmRS_k.labels_
        clst_dict_RS = {}
        for lab in RS_lbs:
            clst_dict_RS[lab] = clst_dict_RS.get(lab, 0) + 1
        for index, label in enumerate(RS_lbs):
            if clst_dict_RS[label] > 1:
                new_CS[label].append(RS[index])  # Append point to cluster
            else:
                new_RS.append(RS[index])  # Append point to new RS
                #print(new_CS)
        clst_sum_CS, clst_ind_CS = cal_clst_stat(new_CS) 
    ds_s, cs_c, cs_s = cal_len(clst_sum_DS, clst_sum_CS)
    rs_s = len(new_RS)
    res_str += "Round 1: " + str(ds_s) + "," + str(cs_c) + "," + str(cs_s) + "," + str(rs_s) + "\n"

#-------------------------------------------------------------------------------------------
#step 7
    thr = 2 * np.sqrt(np.array(first_d).shape[1])
    for index in range(2,6):
        new_RS_2 = []
        new_CS_2 = defaultdict(list)
        #print("Initial CS clusters:", list(clst_sum_CS.keys()))

        if index <= 4:
            assign_data = feature_array[(o_t_f*(index - 1)):(o_t_f*index)]

        else:
            assign_data = feature_array[(o_t_f*(index - 1)):]
        len_RS = len(new_RS)
        for point in assign_data:
                #print(p)
            min_dist = np.inf
            agm_clst = None
            for label, stat in clst_sum_DS.items():
                    # cov = np.diag(stat['variance'])
                    #print(clst_sum_DS)
                distance = mahalanobis_dist(point, stat['centroids'], stat['variance'])
                if distance < min_dist and distance < thr:
                    min_dist = distance
                    agm_clst = label
            if agm_clst is not None:
                clst_sum_DS[agm_clst] = update_cluster_stats(clst_sum_DS[agm_clst], point)
                clst_ind_DS[agm_clst] += fea_dict.get(tuple(point), [])
                clst_pt_ds[agm_clst].append(point)

            else:
                mins = np.inf
                ag_cl = None
                for lb, st in clst_sum_CS.items():
                    dists = mahalanobis_dist(point, st['centroids'], st['variance'])
                    if dists < mins and dists < thr:
                        mins = dists
                        ag_cl = lb
                if ag_cl is not None:
                    clst_sum_CS[ag_cl] = update_cluster_stats(clst_sum_CS[ag_cl], point)
                    clst_ind_CS[ag_cl] += fea_dict.get(tuple(point), [])
                    new_CS[ag_cl].append(point)
                else:
                    new_RS.append(point)

        kmRS_k_2 = None
        if len(new_RS) >= 5 * n_cluster:  # Check if there are enough points to perform K-Means
            kmRS_k_2 = KMeans(n_clusters=5 * n_cluster).fit(np.array(new_RS))
        elif len(new_RS) > 0:
            kmRS_k_2 = KMeans(n_clusters=len(new_RS)).fit(np.array(new_RS))
        else:
            RS_lbs_2 = []
        if kmRS_k_2 is not None:
            RS_lbs_2 = kmRS_k_2.labels_
            clst_RS_2 = {}
            for lab in RS_lbs_2:
                clst_RS_2[lab] = clst_RS_2.get(lab, 0) + 1
            #for ind, label_2 in enumerate(RS_lbs_2):
            for item, label_2 in zip(new_RS, RS_lbs_2):
                if clst_RS_2[label_2] > 1:
                    new_CS[label_2].append(item)
                    if label_2 not in clst_sum_CS:
                         # Append point to cluster
                        clst_sum_CS[label_2] = dict()
                        n = 1
                            #sums = new_RS[ind]
                            #sumsqs = new_RS[ind] ** 2
                        sums = item
                        sumsqs = item ** 2

                        ce = sums / n
                        me = sumsqs / n
                        va = me - np.square(ce)
                        clst_sum_CS[label_2]["N"] = n
                        clst_sum_CS[label_2]["SUM"] = sums
                        clst_sum_CS[label_2]["SUMSQ"] = sumsqs
                        clst_sum_CS[label_2]["centroids"] = ce
                        clst_sum_CS[label_2]["variance"] = va
                        clst_ind_CS[label_2] += fea_dict.get(tuple(item), [])
                    else:
                        clst_sum_CS[label_2] = update_cluster_stats(clst_sum_CS[label_2], item)
                        clst_ind_CS[label_2] += fea_dict.get(tuple(item), [])
                        new_CS[label_2].append(item)
                else:
                    new_RS_2.append(item)
        new_RS = copy.deepcopy(new_RS_2)

        merged_CS = {}
        already_merged = set()
        for lb1 in list(clst_sum_CS.keys()):
            if lb1 in already_merged:
                continue
            merge_into_lb1 = False
            for lb2 in list(clst_sum_CS.keys()):
                if lb1 != lb2 and lb2 not in already_merged:
                        # Calculate Mahalanobis distance between the centroids of lb1 and lb2
                    distance = cluster_cs(clst_sum_CS[lb1], clst_sum_CS[lb2])
                    if distance < thr:
                            # Merge lb2 into lb1
                        if lb1 not in merged_CS:
                            merged_CS[lb1] = clst_sum_CS[lb1].copy()
                        merge_into_lb1 = True
                        already_merged.add(lb2)
                            # Update the cluster properties by summing them
                        merged_CS[lb1]['N'] += clst_sum_CS[lb2]['N']
                        merged_CS[lb1]['SUM'] += clst_sum_CS[lb2]['SUM']
                        merged_CS[lb1]['SUMSQ'] += clst_sum_CS[lb2]['SUMSQ']
                            # Recalculate centroids and variance
                        n = merged_CS[lb1]['N']
                        sums = merged_CS[lb1]['SUM']
                        sumsqs = merged_CS[lb1]['SUMSQ']
                        merged_CS[lb1]['centroids'] = sums / n
                        mes = sumsqs / n
                        merged_CS[lb1]['variance'] = mes - np.square(merged_CS[lb1]['centroids'])
                            # Merge indices
                        clst_ind_CS[lb1].extend(clst_ind_CS[lb2])
                        if lb2 in clst_ind_CS:
                            del clst_ind_CS[lb2]
            if not merge_into_lb1 and lb1 not in already_merged:
                merged_CS[lb1] = clst_sum_CS[lb1]
                already_merged.add(lb1)
        clst_sum_CS = merged_CS



        ds_s_2, cs_c_2, cs_s_2 = cal_len(clst_sum_DS, clst_sum_CS)
        rs_s_2 = len(new_RS)
        if index <= 4:
            res_str += "Round " + str(index) + ": " + str(ds_s_2) + "," + str(cs_c_2) + "," + str(cs_s_2) + "," + str(
                rs_s_2) + "\n"

        if index == 5:
            for cs_ind in list(clst_sum_CS.keys()):
                cs_ds_ind = None
                min_distance = np.inf
                for ds_ind in list(clst_sum_DS.keys()):
                    d = cluster_cs(clst_sum_DS[ds_ind], clst_sum_CS[cs_ind])
                    if d < min_distance:
                        min_distance = d
                        cs_ds_ind = ds_ind
                if min_distance < thr and cs_ds_ind is not None:
                    clst_ind_DS[cs_ds_ind].extend(clst_ind_CS[cs_ind])
                    clst_sum_DS[cs_ds_ind]['N'] += clst_sum_CS[cs_ind]['N']
                    clst_sum_DS[cs_ds_ind]['SUM'] += clst_sum_CS[cs_ind]['SUM']
                    clst_sum_DS[cs_ds_ind]['SUMSQ'] += clst_sum_CS[cs_ind]['SUMSQ']

                    n = clst_sum_DS[cs_ds_ind]['N']
                    sums = clst_sum_DS[cs_ds_ind]['SUM']
                    sumsqs = clst_sum_DS[cs_ds_ind]['SUMSQ']
                    clst_sum_DS[cs_ds_ind]['centroids'] = sums / n
                    me = sumsqs / n
                    clst_sum_DS[cs_ds_ind]['variance'] = me - np.square(clst_sum_DS[cs_ds_ind]['centroids'])
                    del clst_sum_CS[cs_ind]
                    del clst_ind_CS[cs_ind]

            ds_s_f, cs_c_f, cs_s_f = cal_len(clst_sum_DS, clst_sum_CS)
            rs_s_f = len(new_RS)
            res_str += "Round " + str(index) + ": " + str(ds_s_f) + "," + str(cs_c_f) + "," + str(cs_s_f) + "," + str(
                rs_s_f) + "\n"
    #print(res_str)

    res_str += "\nThe clustering results:\n"
    result_dict = {i: -1 for i in range(len(feature_array))}
    for label, indices in clst_ind_DS.items():
        for index in indices:
            result_dict[index] = label

    for label, indices in clst_ind_CS.items():
        for index in indices:
            result_dict[index] = -1  
    sorted_results = sorted(result_dict.items(), key=lambda x: x[0])
    for index, label in sorted_results:
        res_str += f"{index},{label}\n"

    with open(outputs, 'w') as file:
        file.write(res_str)
    end_time = time.time()
    execution = end_time - start_time
    print(f"Duration: {execution}")
    sc.stop()
