import json
from pyspark import SparkContext
import sys
import os
import time

if __name__ == "__main__":
    #check the validation of the command
    if len(sys.argv) != 5:
        print("There is an error for the Usage: python3 task3.py <review_filepath> <business_filepath> <output_filepath_question_a> <output_filepath_question_b>")
        sys.exit(1)
    sc = SparkContext(appName="Task3")

    review_filepath = sys.argv[1]
    business_filepath = sys.argv[2]
    output_filepath_question_a = sys.argv[3]
    output_filepath_question_b = sys.argv[4]

    reviews_rdd_r = (sc.textFile(review_filepath).map(lambda x: json.loads(x)))
    reviews_rdd_b = (sc.textFile(business_filepath).map(lambda x: json.loads(x)))
    city_stars_b = reviews_rdd_b.map(lambda x: (x['business_id'],x['city']))
    city_stars_r = reviews_rdd_r.map(lambda x: (x['business_id'],x['stars']))
    comb_city_star = city_stars_b.join(city_stars_r).map(lambda x: (x[1][0], (x[1][1],1)))
    sum_by_key = comb_city_star.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    average_star = sum_by_key.mapValues(lambda x: x[0] / x[1]).sortBy(lambda x: (-x[1], x[0]))
    result_a = average_star.collect()

    with open(output_filepath_question_a, 'w') as file:
        file.write("city,stars\n")
        for city, stars in result_a:
            file.write(f"{city},{stars}\n")

    start_time_m1 = time.time()
    business_rdd_m1 = sc.textFile(business_filepath).map(lambda x: json.loads(x))
    reviews_rdd_m1 = sc.textFile(review_filepath).map(lambda x: json.loads(x))

    city_b_m1 = business_rdd_m1.map(lambda x: (x['business_id'], x['city']))
    city_r_m1 = reviews_rdd_m1.map(lambda x: (x['business_id'], x['stars']))

    comb_star_m1 = city_b_m1.join(city_r_m1).map(lambda x: (x[1][0], (x[1][1], 1)))
    sum_key_m1 = comb_star_m1.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    average_star_m1 = sum_key_m1.mapValues(lambda x: x[0] / x[1]).collect()

    sort_python = sorted(average_star_m1, key=lambda x: (-x[1], x[0]))[:10]
    end_time_m1 = time.time()
    exe_m1 = end_time_m1 - start_time_m1

    start_time_m2 = time.time()
    business_rdd_m2 = sc.textFile(business_filepath).map(lambda x: json.loads(x))
    reviews_rdd_m2 = sc.textFile(review_filepath).map(lambda x: json.loads(x))

    city_b_m2 = business_rdd_m2.map(lambda x: (x['business_id'], x['city']))
    city_r_m2 = reviews_rdd_m2.map(lambda x: (x['business_id'], x['stars']))

    comb_star_m2 = city_b_m2.join(city_r_m1).map(lambda x: (x[1][0], (x[1][1], 1)))
    sum_key_m2 = comb_star_m2.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    average_star_m2 = sum_key_m1.mapValues(lambda x: x[0] / x[1])
    sort_spark = average_star_m2.sortBy(lambda x: (-x[1], x[0])).take(10)
    end_time_m2 = time.time()
    exe_m2 = end_time_m2 - start_time_m2

    result_b = {
        'm1': exe_m1,
        'm2': exe_m2,
        'reason': ''
    }

    with open(output_filepath_question_b, 'w', encoding='utf-8') as outputs:
        json.dump(result_b, outputs)

    sc.stop()
