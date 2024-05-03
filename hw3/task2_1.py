import sys
from pyspark import SparkContext
import os
import time
import math

def merge_d(x, y):
    return x.union(y)
def merge_dict(x, y):
    for k, v in y.items():
        x[k] = v
    return x


def transform_weight(w, nj):
    w = abs(w)
    if nj <= 2:
        return w
    return w ** 2

def inverse_frequency(n, nj):
    return math.log(n / (nj + 1))


def itembase(use, bus):
    if use not in uni_use:
        return 3
    if bus not in uni_bus:
        return avg_star_u.value.get(use, 3)
    ws = []
    rs = []
    n = len(uni_use)
    for bu in use_dict.value[use]:
        if bu == bus:
            continue

        combuse = list(bus_dict.value[bus].intersection(bus_dict.value[bu]))
        nj = len(combuse)
        fj = inverse_frequency(n, nj)
        if nj == 0:
            w = 1 - abs(- avg_star_b.value[bus] + avg_star_b.value[bu]) / 2
        elif nj == 1:
            w = 1 - abs(org_data.value[bus].get(str(combuse), 3) - avg_star_b.value[bus]
                        - org_data.value[bu].get(str(combuse), 3) + avg_star_b.value[bu] ) / 5
      
        elif nj == 2:
          
            diffs = [(org_data.value[bus].get(user, 3) - avg_star_u.value[user]) - (
                        org_data.value[bu].get(user, 3) - avg_star_u.value[user]) for user in combuse]
            avg_diff = sum(abs(diff) for diff in diffs) / 2
            w = 1 - avg_diff / 5  # normalize the result

        else:
            nu = 0.0
            den_i = 0.0
            den_j = 0.0
            for user in combuse:
                r_i = org_data.value[bu].get(user) - avg_star_b.value.get(bu)
                r_j = org_data.value[bus].get(user) - avg_star_b.value.get(bus)
                nu += r_i *fj * r_j*fj
                den_i += (r_i*fj ) ** 2
                den_j += (r_j*fj ) ** 2

            den = (den_i ** 0.5) * (den_j ** 0.5)
            w = (nu / den) if den != 0 else 0
          
        w_transformed = transform_weight(w, nj)
        r = org_data.value[bu].get(use, avg_star_u.value.get(use, 3))
        rs.append(r)
        ws.append(w_transformed)

    nu = sum(w * r for w, r in zip(ws, rs))
    #print(nu)
    den = sum(abs(w) for w in ws)

    rat = nu / den if den != 0 else 3
    return rat


if __name__ == "__main__":

    #check the validation of the command
    if len(sys.argv) != 4:
        print("There is an error for the Usage: python3 task1.py <train_file_name> <test_file_name> <output_filepath>")
        sys.exit(1)
    sc = SparkContext(appName="Task2_1")
    train = sys.argv[1]
    test = sys.argv[2]
    output = sys.argv[3]

    start_time = time.time()

    sc.setLogLevel("ERROR")
    data = sc.textFile(train)
    header = data.first()
    data_nh = data.filter(lambda line: line != header)
    data_rdd_b = data_nh.map(lambda line: (line.split(',')[1], (float(line.split(',')[2]),1)))
    comb_star_b = data_rdd_b.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    avg_star_b = comb_star_b.mapValues(lambda x: x[0] / x[1]).collectAsMap()

    data_rdd_u = data_nh.map(lambda line: (line.split(',')[0], (float(line.split(',')[2]), 1)))
    comb_star_u = data_rdd_u.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    avg_star_u = comb_star_u.mapValues(lambda x: x[0] / x[1]).collectAsMap()
    avg_star_b = sc.broadcast(avg_star_b)
    avg_star_u = sc.broadcast(avg_star_u)
    org_data = data_nh.map(lambda line: (line.split(',')[1], {line.split(',')[0]: float(line.split(',')[2])})) \
                .reduceByKey(lambda a, b: merge_dict(a, b)).collectAsMap()
    org_data = sc.broadcast(org_data)

    b_rdd = data_nh.map(lambda line: (line.split(',')[1], {line.split(',')[0]}))
    bus_dict = b_rdd.reduceByKey(lambda a, b: merge_d(a, b)).collectAsMap()
    bus_dict = sc.broadcast(bus_dict)
    u_rdd = data_nh.map(lambda line: (line.split(',')[0], {line.split(',')[1]}))
    use_dict = u_rdd.reduceByKey(merge_d).collectAsMap()
    use_dict = sc.broadcast(use_dict)
    val = sc.textFile(test)
    header_val = val.first()
    clean_val = val.filter(lambda row: row != header_val).map(lambda row: row.split(",")).map(lambda row: (row[0], row[1]))
    uni_use = list(use_dict.value.keys())
    uni_bus = list(bus_dict.value.keys())
    cal = clean_val.map(lambda x: (x, itembase(x[0], x[1])))
    results = cal.map(lambda x: ','.join([str(x[0][0]), str(x[0][1]), str(x[1])]))
    collected_results = results.collect()

    with open(output, 'w') as file:
        file.write('user_id,business_id,prediction\n')
        for line in collected_results:
            file.write(f'{line}\n')

    end_time = time.time()
    execution = end_time - start_time
    print(f"Duration: {execution}")
    sc.stop()
