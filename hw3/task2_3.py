import sys
from pyspark import SparkContext
import os
import time
from xgboost import XGBRegressor
import json
import numpy as np
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
        return avg_star_u.value.get(use, 3)  # User's average or fallback

    ws = []
    rs = []
    n = len(uni_use)  # the number of users for inverse frequency calculation

    for bu in use_dict.value[use]:
        if bu == bus:
            continue

        combuse = list(bus_dict.value[bus].intersection(bus_dict.value[bu]))

        nj = len(combuse)
        #print(nj)
        fj = inverse_frequency(n, nj)
        #print(fj)
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
            #print(den)
            w = (nu / den) if den != 0 else 0


        w_transformed = transform_weight(w, nj)
        #w_t = w
        r = org_data.value[bu].get(use, avg_star_u.value.get(use, 3))
        rs.append(r)
        ws.append(w_transformed)
        #ws.append(w_t)
    #print(ws)

    nu = sum(w * r for w, r in zip(ws, rs))
    #print(nu)
    den = sum(abs(w) for w in ws)

    rat = nu / den if den != 0 else 3
    return rat


def modelbase(fold, te, label, feature):
    # ---------------read review json ---------------------
    review_data = sc.textFile(fold + '/review_train.json')
    r_data = review_data.map(lambda line: json.loads(line))
    # keys = r_data.first().keys()
    apt_rdd = r_data.map(lambda x: (x.get('business_id', None),
                                    (float(x.get('useful', None)), float(x.get('funny', None)),
                                     float(x.get('cool', None)), 1)))
    avg_vrb = apt_rdd.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]))
    # take the average of the business_id in review_train
    avg_vrb = avg_vrb.mapValues(lambda x: (x[0] / x[3], x[1] / x[3], x[2] / x[3])).collectAsMap()
    # {business_id: {useful, funny, cool}}
    buss_data = sc.textFile(fold + '/business.json')
    b_data = buss_data.map(lambda line: json.loads(line))
    # dict_keys(['business_id', 'name', 'neighborhood', 'address', 'city', 'state', 'postal_code', 'latitude', 'longitude', 'stars', 'review_count', 'is_open', 'attributes', 'categories', 'hours'])
    apt_b = b_data.map(lambda x: (x.get('business_id', None), (
    float(x.get('stars', None)), float(x.get('review_count', None)), float(x.get('is_open', None))))).collectAsMap()
    # {business_id: {stars, review_count, is_open}}
    comb_b = {}
    # combine these two business id union
    buss_ids = set(avg_vrb.keys()) | set(apt_b.keys())
    for business_id in buss_ids:
        # initialize all variables in these two with None
        ufc = avg_vrb.get(business_id, (None, None, None))
        sri = apt_b.get(business_id, (None, None, None))
        comb_b[business_id] = {
            'useful': ufc[0],
            'funny': ufc[1],
            'cool': ufc[2],
            'stars': sri[0],
            'review_count': sri[1],
            'is_open': sri[2]
        }

    user_data = sc.textFile(fold + '/user.json')
    u_data = user_data.map(lambda line: json.loads(line))
    apt_u = u_data.map(lambda x: (x.get('user_id', None),
                                  (float(x.get('average_stars', None)), float(x.get('review_count', None)),
                                   float(x.get('useful', None))))).collectAsMap()
    # {user: avg_star, review, useful_u}
    comb_f = []
    for user, business in feature:
        entry = [None] * 9
        if user in apt_u:
            use = apt_u[user]
            entry[0] = use[0]  # avg_star
            entry[1] = use[1]  # review
            entry[2] = use[2]  # useful_u

        if business in comb_b:
            bd = comb_b[business]
            entry[3] = bd['useful']
            entry[4] = bd['funny']
            entry[5] = bd['cool']
            entry[6] = bd['stars']
            entry[7] = bd['review_count']
            entry[8] = bd['is_open']

        # comb_f.append((user, business, entry))
        comb_f.append(entry)
    # print(comb_f)
    X_train = np.array(comb_f, dtype='float32')
    Y_train = np.array(label, dtype='float32')

    val_t = te.collect()

    comb_t = []
    for user, business in val_t:
        ent = [None] * 9
        if user in apt_u:
            use = apt_u[user]
            ent[0] = use[0]  # avg_star
            ent[1] = use[1]  # review
            ent[2] = use[2]  # useful_u

        if business in comb_b:
            bd = comb_b[business]
            ent[3] = bd['useful']
            ent[4] = bd['funny']
            ent[5] = bd['cool']
            ent[6] = bd['stars']
            ent[7] = bd['review_count']
            ent[8] = bd['is_open']

        comb_t.append(ent)
    X_test = np.array(comb_t, dtype='float32')
    model = XGBRegressor()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    return val_t, Y_pred

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("There is an error for the Usage: python3 task1.py <folder_name> <test_file_name> <output_filepath>")
        sys.exit(1)
    sc = SparkContext(appName="Task2_3")
    folder = sys.argv[1]
    test = sys.argv[2]
    output = sys.argv[3]

    start_time = time.time()
    sc.setLogLevel("ERROR")
    train_data = sc.textFile(folder + '/yelp_train.csv')
    header = train_data.first()
    data_nh = train_data.filter(lambda line: line != header).map(lambda row: row.split(","))
    data_rdd_b = data_nh.map(lambda line: (line[1], (float(line[2]), 1)))
    comb_star_b = data_rdd_b.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    avg_star_b = comb_star_b.mapValues(lambda x: x[0] / x[1]).collectAsMap()
    data_rdd_u = data_nh.map(lambda line: (line[0], (float(line[2]), 1)))
    comb_star_u = data_rdd_u.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    avg_star_u = comb_star_u.mapValues(lambda x: x[0] / x[1]).collectAsMap()
    avg_star_b = sc.broadcast(avg_star_b)
    avg_star_u = sc.broadcast(avg_star_u)
    org_data = data_nh.map(lambda line: (line[1], {line[0]: float(line[2])})) \
        .reduceByKey(lambda a, b: merge_dict(a, b)).collectAsMap()
    org_data = sc.broadcast(org_data)
    b_rdd = data_nh.map(lambda line: (line[1], {line[0]}))
    bus_dict = b_rdd.reduceByKey(lambda a, b: merge_d(a, b)).collectAsMap()
    bus_dict = sc.broadcast(bus_dict)
    u_rdd = data_nh.map(lambda line: (line[0], {line[1]}))
    use_dict = u_rdd.reduceByKey(merge_d).collectAsMap()
    use_dict = sc.broadcast(use_dict)
    features = data_nh.map(lambda x: (x[0], x[1])).collect()  # (user_id, business_id)
    labels = data_nh.map(lambda x: x[2]).collect()
    val = sc.textFile(test)
    header_val = val.first()
    clean_val = val.filter(lambda row: row != header_val).map(lambda row: row.split(",")).map(
        lambda row: (row[0], row[1]))
    uni_use = list(use_dict.value.keys())
    uni_bus = list(bus_dict.value.keys())
    cal = clean_val.map(lambda x: itembase(x[0], x[1])).collect()
    #model base result
    val_var, prediction_model = modelbase(folder, clean_val, labels, features)
    #print(prediction_model)
    alpha = 0.01
    final_score = alpha * np.array(cal, dtype='float32') + (1-alpha)* np.array(prediction_model, dtype='float32')

    with open(output, 'w') as file:
        file.write('user_id,business_id,prediction\n')
        for ((user_id, business_id), prediction) in zip(val_var, final_score):
            line = f'{user_id},{business_id},{prediction}'
            file.write(f'{line}\n')

    end_time = time.time()
    print('Duration: ', end_time - start_time)
    #RMSE: 0.9858999665858088 0.05
    #RMSE: 0.9863866718505088 0.08
    #RMSE: 0.9856799493270916 0.03
    #RMSE: 0.9855435649597785 0.01
    sc.stop()
