import json
from datetime import datetime
import sys
from pyspark import SparkContext
import os


def review_info(review_rdd):

    total_num = review_rdd.count()

    review_2018 = review_rdd.filter(lambda review: datetime.strptime(json.loads(review)['date'], "%Y-%m-%d %H:%M:%S").year == 2018).count()

    unique_user = review_rdd.map(lambda review: json.loads(review)['user_id']).distinct().count()

    user_count = review_rdd.map(lambda x: [json.loads(x)['user_id'], 1]).reduceByKey(lambda a, b: a + b).sortBy(lambda x: (-x[1], x[0])).take(10)

    unique_buss = review_rdd.map(lambda review: json.loads(review)['business_id']).distinct().count()

    buss_count = review_rdd.map(lambda review: [json.loads(review)['business_id'], 1]).reduceByKey(lambda a, b: a + b).sortBy(lambda x: (-x[1], x[0])).take(10)


    final_result = {
        "n_review": total_num,
        "n_review_2018": review_2018,
        "n_user": unique_user,
        "top10_user": user_count,
        "n_business": unique_buss,
        "top10_business": buss_count
    }

    return final_result


if __name__ == "__main__":
    #check the validation of the command
    if len(sys.argv) != 3:
        print("There is an error for the Usage: python3 task1.py <review_filepath> <output_filepath>")
        sys.exit(1)

    sc = SparkContext(appName="Task1")

    review_filepath = sys.argv[1]
    output_filepath = sys.argv[2]

    reviews_rdd = sc.textFile(review_filepath)

    final_result = review_info(reviews_rdd)

    with open(output_filepath, 'w', encoding='utf-8') as outputs:
        json.dump(final_result, outputs)

    sc.stop()
