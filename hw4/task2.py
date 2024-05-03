import sys
from pyspark import SparkContext
import os
import time
from itertools import combinations
from collections import deque

def GNA(group, nodes):
    b_v = {}
    for root in nodes:
        parent = {}
        level = {}
        q = deque([root])
        # print(queues)
        visit = set()
        visit.add(root)
        level[root] = 0
        n_stp = {root: 1}
        path = [root]
        while q:
            current = q.popleft()
            for nb in group.get(current, []):
                if nb not in visit:#if not visit neighbour
                    level[nb] = level[current] + 1
                    visit.add(nb)
                    q.append(nb)
                    path.append(nb)
                    if nb not in parent.keys():
                        parent[nb] = set()
                    parent[nb].add(current)
                    n_stp[nb] = n_stp.get(nb, 0) + n_stp.get(current, 0)
                elif nb in visit and level[nb] == level[current] + 1:
                    parent[nb].add(current)
                    # refresh the number of shortest paths
                    n_stp[nb] += n_stp[current]

        n_w = {}
        e_w = {}
        rev_path = list(reversed(path))
        for rv in rev_path:
            n_w[rv] = 1
        for a in rev_path:
            for b in parent.get(a, []):
                w = n_w.get(a, 1) * (n_stp.get(b, 1) / n_stp.get(a, 1))
                n_w[b] = n_w.get(b, 1) + w
                edge_n = tuple((a, b)) if a < b else tuple((b, a))

                e_w[edge_n] = e_w.get(edge_n, 0) + w

        for key, value in e_w.items():
            b_v[key] = b_v.get(key, 0) + value / 2

    b = sorted(b_v.items(), key=lambda x: (-x[1], x[0]))
    return b

def cal_mod(c, edge, mx, k_nn):
    modularity = 0.0
    for community in comits:
        for i in community:
            for j in community:
                a_ij = 1 if j in edges_g.get(i, []) else 0.0
                modularity += (a_ij - (k_n[i] * k_n[j]) / (2.0 * m))
    return modularity / (2.0 * m)

if __name__ == "__main__":

    #check the validation of the command
    if len(sys.argv) != 5:
        print("There is an error for the Usage: python3 task2.py <threshold> <input_filepath> <betweenness_output_filepath> <community_output_filepath>")
        sys.exit(1)

    sc = SparkContext(appName="Task2")

    thr = int(sys.argv[1])
    input_file = sys.argv[2]
    between_output = sys.argv[3]
    community_output = sys.argv[4]

    start_time = time.time()
    sc.setLogLevel("ERROR")
    csv_rdd = sc.textFile(input_file)
    header = csv_rdd.first()
    data_nh = (csv_rdd.filter(lambda x: x != header).map(lambda x: x.split(","))
               .map(lambda x: (x[0], {x[1]}))
               .reduceByKey(lambda a, b: a.union(b)))
    data_pre = data_nh.filter(lambda x: len(x[1]) >= thr).collectAsMap()
    uni_user = list(data_pre.keys())
    combuser = list(combinations(uni_user, 2))
    filter_p = sc.parallelize(combuser).filter(
        lambda users: len(data_pre[users[0]].intersection(data_pre[users[1]])) >= thr)
    edges_n = filter_p.flatMap(lambda users: [(users[1], users[0]), (users[0], users[1])])
    nodes = filter_p.flatMap(lambda users: [users[0], users[1]]).distinct().map(lambda user: user)
    n_list = list(nodes.collect())
    edges_kv = edges_n.map(lambda x: (x[0], {x[1]}))
    edges_gr = edges_kv.reduceByKey(lambda a, b: a.union(b))
    edges_g = edges_gr.collectAsMap()
    betweens = GNA(edges_g, n_list)
    #print(betweens)
    output_1 = [(pair, round(between, 5)) for pair, between in betweens]
    with open(between_output, "w") as f:
        for users, value in output_1:
            f.write(f"{users},{value}\n")
    m = len(betweens) #determine m
    l_g = edges_gr.map(lambda x: (x[0], len(x[1]))) #determine the k_i,k_j
    k_n = l_g.collectAsMap() #make as a dictionary

    max_mod = -float('inf')
    final_com = []

    while betweens:
        edge_rm, _ = betweens[0]
        #remove the nodes from the original dictionary
        edges_g[edge_rm[0]].discard(edge_rm[1])
        edges_g[edge_rm[1]].discard(edge_rm[0])

        comits = []
        ori_n = set(n_list)
        while ori_n:
            root = ori_n.pop()
            queue = deque([root])
            visit_ag = set([root])
            com = [root]
            while queue:
                current = queue.popleft()
                for nb in edges_g.get(current, []):
                    if nb not in visit_ag:
                        visit_ag.add(nb)
                        ori_n.discard(nb)
                        queue.append(nb)
                        com.append(nb)
            #struct the splited communities
            comits.append(sorted(com))

        cur_mod = cal_mod(comits, edges_g, m, k_n)

        if cur_mod > max_mod:
            max_mod = cur_mod
            final_com = [list(comm) for comm in comits]  # Avoid deep copy by reconstructing lists

        betweens = GNA(edges_g, n_list)
    result = sorted(final_com, key=lambda x: (len(x), x[0]))
    with open(community_output, "w") as f:
        for use in result:
            com = ', '.join([f"'{u}'" for u in use])
            f.write(com + "\n")

    #---------------------------------------------------------------------------

    end_time = time.time()
    execution = end_time - start_time
    print(f"Duration: {execution}")
    sc.stop()
