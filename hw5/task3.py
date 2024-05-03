from blackbox import BlackBox
import sys
from pyspark import SparkContext
import os
import time
import random

if __name__ == "__main__":

    inputs = sys.argv[1]
    stream = int(sys.argv[2])
    asks = int(sys.argv[3])
    outputs = sys.argv[4]

    start_time = time.time()
    #sc.setLogLevel("ERROR")
    random.seed(553)
    bx = BlackBox()
    result_header = "seqnum,0_id,20_id,40_id,60_id,80_id\n"
    user_list = []
    total = 0
    size = 100
    #result_h = ""
    for i in range(asks):
        # Simulate streaming users from the BlackBox
        users_stream = bx.ask(inputs, stream)
        for user in users_stream:
            total += 1
            if len(user_list) < size:
                user_list.append(user)

            else:
                if random.random() < size / total:
                    user_list[random.randint(0, size - 1)] = user
            if total % size == 0:
                elem = [str(total), str(user_list[0]), str(user_list[20]), str(user_list[40]), str(user_list[60]),
                        str(user_list[80])]
                result_header += ','.join(elem) + '\n'
    with open(outputs, 'w') as file:
        file.write(result_header)
    end_time = time.time()
    execution = end_time - start_time
    print(f"Duration: {execution}")
