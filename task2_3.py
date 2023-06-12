import math
from pyspark import SparkContext
import time
import sys
import json
import xgboost as xgb

def predict(user,business): # default rate is 3.0
    if user not in user_dict2.keys():
        return [(user, business), 3.0]
    if business not in business_dict2.keys():
        return [(user, business), 3.0]
    rated_items=user_dict2[user]
    w={} # store Pearson correlations
    for item in rated_items: # item in business_dict
        if item != business:
            U=list(set(business_dict2[item]).intersection(set(business_dict2[business]))) # users who rated both i&j
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

def get_feature(u,b):
    if u in user_dict1 and b in business_dict1:
        u_reviewcount = user_dict1[u][0]
        b_reviewcount = business_dict1[b][0]
        u_avgstar = user_dict1[u][1]
        b_avgstar = business_dict1[b][1]
        feature=[u_reviewcount, b_reviewcount, u_avgstar, b_avgstar]
    else:
        feature=[2,5,3.0,3.0]
    return feature

if __name__ == '__main__':
    start = time.time()
    # execution format: spark-submit task2_3.py <folder_path> <test_file_name> <output_file_name>
    folder = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]
    train_file = folder + "/yelp_train.csv"
    user_file = folder + "/user.json"
    business_file = folder + "/business.json"

    sc = SparkContext.getOrCreate()
    sc.setLogLevel('Error')

    # model_based
    RDD0 = sc.textFile(train_file).map(lambda x: x.split(",")).filter(lambda x: x[0] != 'user_id')  # remove the header line in csv
    train_set = RDD0.map(lambda x: ((x[0], x[1]), float(x[2])))  # {(user_id,business_id),star}
    RDD1 = sc.textFile(test_file).map(lambda x: x.split(",")).filter(lambda x: x[0] != 'user_id')
    val_set = RDD1.map(lambda x: (x[0], x[1]))  # (user_id,business_id)
    user_info1 = sc.textFile(user_file).map(lambda x: json.loads(x))
    user_feature = user_info1.map(lambda x: (x["user_id"], (int(x["review_count"]), float(x["average_stars"]))))  # select the average stars and the number of reviews as features
    business_info1 = sc.textFile(business_file).map(lambda x: json.loads(x))
    business_feature = business_info1.map(lambda x: (x["business_id"], (int(x["review_count"]), float(x["stars"]))))
    user_dict1 = user_feature.collectAsMap()
    business_dict1 = business_feature.collectAsMap()

    # item_based
    item_avg = RDD0.map(lambda x: (x[1], float(x[2]))).groupByKey().map(lambda y: (y[0], sum(y[1]) / len(y[1])))  # (business_id, average_star)
    user_info2 = RDD0.map(lambda x: (x[0], x[1])).groupByKey().map(lambda y: (y[0], list(y[1])))  # (user_id,[business_ids])
    business_info2 = RDD0.map(lambda x: (x[1], x[0])).groupByKey().map(lambda y: (y[0], list(y[1])))  # (business_id,[user_ids])
    user_dict2 = user_info2.collectAsMap()
    business_dict2 = business_info2.collectAsMap()
    rate_dict = train_set.collectAsMap()
    itemavg_dict = item_avg.collectAsMap()
    result1 = RDD1.map(lambda x: predict(x[0],x[1])).collect()

    # model_based
    x_train = train_set.map(lambda x: get_feature(x[0][0], x[0][1])).collect()
    y_train = train_set.map(lambda x: x[1]).collect()
    x_val = val_set.map(lambda x: get_feature(x[0], x[1])).collect()
    xgb_model = xgb.XGBRegressor(objective='reg:linear', n_estimators=100, n_jobs=-1)
    xgb_model.fit(x_train, y_train)
    y_pred = xgb_model.predict(x_val)
    result2 = list(zip(val_set.collect(), y_pred))  # [((user_id,business_id),pred_star)]

    result = []  # combine 2 recommendation systems with weighted average
    for r in range(len(result2)):
        if user_dict1[result2[r][0][0]][0] > 5 or business_dict1[result2[r][0][1]][0] > 10:  # more trustworthy with more review counts
            result.append((result2[r][0], 0.2 * result1[r][1] + 0.8 * result2[r][1]))  # larger weight on model_based
        else:
            result.append((result2[r][0], 0.5 * result1[r][1] + 0.5 * result2[r][1]))  # equal weight

    with open(output_file, 'w+') as f:
        f.write("user_id, business_id, prediction\n")
        for row in result:
            f.write(row[0][0] + ',' + row[0][1] + ',' + str(row[1]) + '\n')
        f.close()
    end = time.time()
    print("Duration:", end - start)