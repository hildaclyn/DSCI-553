import sys
from pyspark import SparkContext
import os
import time
from xgboost import XGBRegressor
import json
import numpy as np

if __name__ == "__main__":

    #check the validation of the command
    if len(sys.argv) != 4:
        print("There is an error for the Usage: python3 task1.py <path_name> <test_file_name> <output_filepath>")
        sys.exit(1)
    sc = SparkContext(appName="Task2_2")
    folder = sys.argv[1]
    test = sys.argv[2]
    output = sys.argv[3]

    start_time = time.time()

    sc.setLogLevel("ERROR")
    train_data = sc.textFile(folder + '/yelp_train.csv')
    # read the header and remove it
    header = train_data.first()
    data_nh = train_data.filter(lambda line: line != header).map(lambda row: row.split(","))
    features = data_nh.map(lambda x: (x[0], x[1])).collect()  # (user_id, business_id)
    #print(features)
    labels = data_nh.map(lambda x: x[2]).collect() # y train set
    #---------------read review json ---------------------
    review_data = sc.textFile(folder + '/review_train.json')
    r_data = review_data.map(lambda line: json.loads(line))
    apt_rdd = r_data.map(lambda x: (x.get('business_id', None),
                                   (float(x.get('useful', None)), float(x.get('funny', None)), float(x.get('cool', None)), 1)))
    avg_vrb = apt_rdd.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]))
    avg_vrb = avg_vrb.mapValues(lambda x: (x[0] / x[3], x[1] / x[3], x[2] / x[3])).collectAsMap()

    buss_data = sc.textFile(folder + '/business.json')
    b_data = buss_data.map(lambda line: json.loads(line))
    #dict_keys(['business_id', 'name', 'neighborhood', 'address', 'city', 'state', 'postal_code', 'latitude', 'longitude', 'stars', 'review_count', 'is_open', 'attributes', 'categories', 'hours'])
    apt_b = b_data.map(lambda x: (x.get('business_id', None),(float(x.get('stars', None)), float(x.get('review_count', None)), float(x.get('is_open', None))))).collectAsMap()

    comb_b = {}
    # combine these two business id union
    buss_ids = set(avg_vrb.keys()) | set(apt_b.keys())

    for business_id in buss_ids:
        #initialize all variables in these two with None
        ufc = avg_vrb.get(business_id, (None, None, None))
        sri = apt_b.get(business_id,(None, None, None))

        comb_b[business_id] = {
            'useful': ufc[0],
            'funny': ufc[1],
            'cool': ufc[2],
            'stars': sri[0],
            'review_count': sri[1],
            'is_open': sri[2]
        }

    user_data = sc.textFile(folder + '/user.json')
    u_data = user_data.map(lambda line: json.loads(line))
    apt_u = u_data.map(lambda x: (x.get('user_id', None),
                                  (float(x.get('average_stars', None)), float(x.get('review_count', None)),
                                   float(x.get('useful', None))))).collectAsMap()


    comb_f = []

    for user, business in features:
        # Initialize the features with None to handle missing data
        entry = [None] * 9
        if user in apt_u:
            use = apt_u[user]
            entry[0] = use[0]  # avg_star
            entry[1] = use[1]  # review
            entry[2] = use[2]  # useful_u

        # If business exists in comb_b, update the relevant features
        if business in comb_b:
            bd = comb_b[business]
            entry[3] = bd['useful']
            entry[4] = bd['funny']
            entry[5] = bd['cool']
            entry[6] = bd['stars']
            entry[7] = bd['review_count']
            entry[8] = bd['is_open']

        comb_f.append(entry)
    #print(comb_f)
    X_train = np.array(comb_f, dtype='float32')
    Y_train = np.array(labels, dtype='float32')

    val = sc.textFile(test)
    val_h = val.first()
    val_nh = val.filter(lambda row: row != val_h).map(lambda row: row.split(","))
    val_t = val_nh.map(lambda x: (x[0], x[1])).collect()

    comb_t = []
    for user, business in val_t:
        # initialize with None to handle missing data
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
#----------create a regression -----------------------
    model = XGBRegressor()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    #print(Y_pred[:5])

    with open(output, 'w') as file:
        file.write('user_id,business_id,prediction\n')

        for ((user_id, business_id), prediction) in zip(val_t, Y_pred):
            line = f'{user_id},{business_id},{prediction}'
            file.write(f'{line}\n')

    end_time = time.time()
    print('Duration: ', end_time - start_time)
    
    sc.stop()
