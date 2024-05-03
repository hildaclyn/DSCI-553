from blackbox import BlackBox
import sys
from pyspark import SparkContext
import os
import time
import random
import binascii

def myhashs(s):
    p = int(1e5 + 3)
    user_int = int(binascii.hexlify(s.encode('utf8')), 16)
    result = []
    for a, b in hash_a_b:
        # Apply each hash function to the user ID
        hash_val = ((a * user_int + b) % p) % bit_len
        result.append(hash_val)
    return result

if __name__ == "__main__":

    sc = SparkContext(appName="Task1")

    #capture the parameter from linux command
    inputs = sys.argv[1]
    stream = int(sys.argv[2])
    asks = int(sys.argv[3])
    outputs = sys.argv[4]

    start_time = time.time()
    sc.setLogLevel("ERROR")
    #random.seed(553)
    bx = BlackBox()
    bit_len = 69997
    bit_ar = [0] * bit_len

    h_num = 30
    hash_a_b = [(random.randint(1, bit_len), random.randint(0, bit_len)) for i in range(h_num)]

    result_header = "Time,FPR\n"
    total = 0
    p_user = set()


    for i in range(asks):
        streamusers = bx.ask(inputs, stream)
        user_s = sc.parallelize(streamusers)
        user_h = user_s.map(lambda user: (user, myhashs(user)))
        results = user_h.collect()
        FP = 0

        for user, hashes in results:
            new = not all(bit_ar[h] for h in hashes)
            if new:
                #a new user, set the bits
                for h in hashes:
                    bit_ar[h] = 1
            else:
                if user not in p_user:
                # all bits are set, repeated situation
                    FP += 1
            p_user.add(user)
            total += 1

        fpr = FP / stream
        elem = [str(i), str(fpr)]
        result_header += ','.join(elem) + '\n'
    with open(outputs, 'w') as file:
        file.write(result_header)
    end_time = time.time()
    execution = end_time - start_time
    print(f"Duration: {execution}")
    sc.stop()
