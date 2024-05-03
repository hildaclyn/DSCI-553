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
        hash_val = ((a * user_int + b) % p) % bit_len
        result.append(int(hash_val))
    return result

def count_trails(bina):
    binary = bin(bina)
    trail_len = len(binary)
    rm_zero = len(binary.rstrip('0'))
    out = trail_len - rm_zero
    return out

if __name__ == "__main__":

    #check the validation of the command
    if len(sys.argv) != 5:
        print("There is an error for the Usage: python3 task2.py <input_filename> stream_size num_of_asks <output_filename>")
        sys.exit(1)

    sc = SparkContext(appName="Task2")

    inputs = sys.argv[1]
    stream = int(sys.argv[2])
    asks = int(sys.argv[3])
    outputs = sys.argv[4]

    start_time = time.time()
    sc.setLogLevel("ERROR")
    #random.seed(553)
    bx = BlackBox()
    result_header = "Time,Ground Truth,Estimation\n"
    h_num = 30
    ground = 0
    est_sum = 0
    bit_len = 2345
    h = 30
    hash_a_b = [(random.randint(1, bit_len), random.randint(0, bit_len)) for i in range(h)]
    for i in range(asks):
        streamusers = bx.ask(inputs, stream)
        user_s = sc.parallelize(streamusers)
        user_h = user_s.map(lambda user: (user, myhashs(user)))
        user_binary = user_h.map(lambda x: (x[0], [count_trails(el) for el in x[1]]))
        user_r = user_binary.collect()
        max_zeros = [0] * h_num
        est = 0
        e_u = set()

        for user, hashs in user_r:
            e_u.add(user)
            for ind, zeros in enumerate(hashs):
                if zeros > max_zeros[ind]:
                    max_zeros[ind] = zeros
        #print(max_zeros)
        est += sum([2 ** zeros for zeros in max_zeros])
        est_p = est // h_num
        ground += len(e_u)
        est_sum += est_p

        elem = [str(i), str(len(e_u)), str(est_p)]
        result_header += ','.join(elem) + '\n'
    #print(est_sum/ground)

    with open(outputs, 'w') as file:
        file.write(result_header)
    end_time = time.time()
    execution = end_time - start_time
    print(f"Duration: {execution}")
    sc.stop()
