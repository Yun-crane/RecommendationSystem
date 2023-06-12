from pyspark import SparkContext
import time
import sys
import sympy
import random
from itertools import combinations



def get_signature(user_ids):
    signs=[]
    for j in range(hash_num):
        signs.append(min(list(map(lambda y: ((a_list[j]*y+b_list[j])%p_list[j])%(2*l),user_ids)))) # m=2l, min-hash
    return signs


def split_band(sgn):
    splits=[]
    for q in range(b):
        band=sgn[1][q*r:(q+1)*r] # elements in each band
        splits.append(((q,tuple(band)),[sgn[0]])) # key is (band_index,band_data) pair, value is business_id
    return splits


def Jaccard(pair):
    bus1=bus_info[pair[0]]
    bus2=bus_info[pair[1]]
    sim=len(set(bus1).intersection(set(bus2)))/len(set(bus1).union(set(bus2)))
    return sim


if __name__ == '__main__':
    start = time.time()
    # execution format: spark-submit task1.py <input_file_name> <output_file_name>
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    sc = SparkContext.getOrCreate()
    RDD0 = sc.textFile(input_file).map(lambda x: x.split(",")).filter(lambda x: x[0] != 'user_id')  # remove the header line in csv
    RDD1 = RDD0.map(lambda x: (x[1], x[0]))  # only retain the business_id and the user_id
    user_list = sorted(list(set(RDD1.map(lambda x: x[1]).collect())))  # sort and remove duplicate in user_id
    user_index = {}
    for i in range(len(user_list)):
        user_index[user_list[i]] = i
    RDD2 = RDD1.map(lambda x: (x[0], user_index[x[1]]))  # transfer the format of user_id
    bus_info1 = RDD2.groupByKey().map(lambda x: (x[0], sorted(list(set(x[1])))))  # (business_id,<iterable>) => (business_id,[user_ids]) remove duplication
    bus_info2 = bus_info1.sortByKey()
    l = len(user_list)
    hash_num = 150
    p_list = random.sample(list(sympy.primerange(l, 2 * l)), hash_num)  # p is a prime number > l
    a_list = random.sample([k for k in range(l)], hash_num)  # a<l
    b_list = random.sample([k for k in range(l)], hash_num)  # b<l
    signatures = bus_info2.mapValues(lambda ids: get_signature(ids))
    r = 3  # number of rows per band
    b = 50  # number of bands
    band_result = signatures.flatMap(lambda z: split_band(z)).reduceByKey(lambda x, y: x + y)  # find identical band and group the business_id
    candidates_business = band_result.filter(lambda m: len(m[1]) > 1).flatMap(lambda z: list(combinations(z[1], 2))).distinct()
    bus_info = bus_info2.collectAsMap()  # key is business_id, value is user_id list
    honest_result = candidates_business.map(lambda x: tuple(sorted(list(x)))).map(lambda y: (y[0], y[1], Jaccard(y))).filter(lambda z: z[2] >= 0.5).sortBy(lambda u: (u[0], u[1])).collect()

    with open(output_file, 'w+') as f:
        f.write("business_id_1, business_id_2, similarity\n")
        for row in honest_result:
            f.write(row[0]+','+row[1]+','+str(row[2])+'\n')
        f.close()
    end = time.time()
    print("Duration:", end - start)