import math
from pyspark import SparkContext
import time
import sys



def predict(user,business): # default rate is 3.0
    if user not in user_dict.keys():
        return [(user, business), 3.0]
    if business not in business_dict.keys():
        return [(user, business), 3.0]
    rated_items=user_dict[user]
    w={} # store Pearson correlations
    for item in rated_items: # item in business_dict
        if item != business:
            U=list(set(business_dict[item]).intersection(set(business_dict[business]))) # users who rated both i&j
            numerator=0
            denominator1=0
            denominator2=0
            if len(U)>1:
                for u in U:
                    numerator += (rate_dict[(u,item)]-itemavg_dict[item])*(rate_dict[(u,business)]-itemavg_dict[business])
                    denominator1 += (rate_dict[(u,item)]-itemavg_dict[item])**2
                    denominator2 += (rate_dict[(u,business)] - itemavg_dict[business])**2
                if denominator1==0 or denominator2==0:
                    w[item]=0
                elif numerator<0: # discard the negative correlations
                    None
                else:
                    w[item]=numerator/(math.sqrt(denominator1)*math.sqrt(denominator2))
            else: # deal with no common users
                dif = abs(itemavg_dict[item]-itemavg_dict[business])
                if dif <= 1: # similar items
                    w[item]=1
                elif 1 < dif <= 2: # semi-similar
                    w[item]=0.5
                else:
                    w[item]=0
        else:
            w[item]=1
    if len(w.keys())==0:
        return [(user, business), 3.0]
    else:
        cand=sorted([(cor,itm) for itm,cor in w.items()], reverse=True) # (correlation w, item n)
        nmr=0
        dmr=0
        for c in cand[:100]: # N=100
            nmr+=rate_dict[(user,c[1])]*c[0]
            dmr+=abs(c[0])
        if dmr==0:
            return [(user, business), 3.0]
        else:
            return [(user, business), nmr/dmr]

if __name__ == '__main__':
    start = time.time()
    # execution format: spark-submit task2_1.py <train_file_name> <test_file_name> <output_file_name>
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    sc = SparkContext.getOrCreate()

    RDD0 = sc.textFile(train_file).map(lambda x: x.split(",")).filter(lambda x: x[0] != 'user_id')  # remove the header line in csv
    RDD1 = RDD0.map(lambda x: ((x[0], x[1]), float(x[2])))  # {(user_id,business_id),star}
    item_avg = RDD0.map(lambda x: (x[1], float(x[2]))).groupByKey().map(lambda y: (y[0], sum(y[1]) / len(y[1])))  # (business_id, average_star)
    user_info = RDD0.map(lambda x: (x[0], x[1])).groupByKey().map(lambda y: (y[0], list(y[1])))  # (user_id,[business_ids])
    business_info = RDD0.map(lambda x: (x[1], x[0])).groupByKey().map(lambda y: (y[0], list(y[1])))  # (business_id,[user_ids])
    user_dict = user_info.collectAsMap()
    business_dict = business_info.collectAsMap()
    rate_dict = RDD1.collectAsMap()
    itemavg_dict = item_avg.collectAsMap()
    valdata = sc.textFile(test_file).map(lambda x: x.split(",")).filter(lambda x: x[0]!='user_id')
    result=valdata.map(lambda x: predict(x[0],x[1])).collect()
    with open(output_file, 'w+') as f:
        f.write("user_id, business_id, prediction\n")
        for row in result:
            f.write(row[0][0] + ',' + row[0][1] + ',' + str(row[1]) + '\n')
        f.close()
    end = time.time()
    print("Duration:", end - start)



