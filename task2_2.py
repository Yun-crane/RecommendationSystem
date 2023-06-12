import math
from pyspark import SparkContext
import time
import sys
import json
import xgboost as xgb


def get_feature(u,b):
    u_reviewcount = user_dict[u][0]
    b_reviewcount = business_dict[b][0]
    u_avgstar = user_dict[u][1]
    b_avgstar = business_dict[b][1]
    feature=[u_reviewcount, b_reviewcount, u_avgstar, b_avgstar]
    return feature


if __name__ == '__main__':
    start = time.time()
    # execution format: spark-submit task2_2.py <folder_path> <test_file_name> <output_file_name>
    folder = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]
    train_file=folder+"/yelp_train.csv"
    user_file=folder+"/user.json"
    business_file=folder+"/business.json"

    sc = SparkContext.getOrCreate()
    sc.setLogLevel('Error')

    RDD0 = sc.textFile(train_file).map(lambda x: x.split(",")).filter(lambda x: x[0] != 'user_id')  # remove the header line in csv
    train_set = RDD0.map(lambda x: ((x[0], x[1]), float(x[2])))  # {(user_id,business_id),star}
    RDD1 = sc.textFile(test_file).map(lambda x: x.split(",")).filter(lambda x: x[0] != 'user_id')
    val_set = RDD1.map(lambda x: (x[0], x[1])) # (user_id,business_id)
    user_info=sc.textFile(user_file).map(lambda x: json.loads(x))
    user_feature=user_info.map(lambda x: (x["user_id"],(int(x["review_count"]),float(x["average_stars"])))) # select the average stars and the number of reviews as features
    business_info=sc.textFile(business_file).map(lambda x: json.loads(x))
    business_feature=business_info.map(lambda x: (x["business_id"],(int(x["review_count"]),float(x["stars"]))))
    user_dict=user_feature.collectAsMap()
    business_dict=business_feature.collectAsMap()

    x_train=train_set.map(lambda x: get_feature(x[0][0],x[0][1])).collect()
    y_train=train_set.map(lambda x: x[1]).collect()
    x_val=val_set.map(lambda x: get_feature(x[0],x[1])).collect()
    xgb_model=xgb.XGBRegressor(objective = 'reg:linear', n_estimators=100, n_jobs=-1)
    xgb_model.fit(x_train,y_train)
    y_pred=xgb_model.predict(x_val)

    result=list(zip(val_set.collect(),y_pred)) # [((user_id,business_id),pred_star)]
    with open(output_file, 'w+') as f:
        f.write("user_id, business_id, prediction\n")
        for row in result:
            f.write(row[0][0] + ',' + row[0][1] + ',' + str(row[1]) + '\n')
        f.close()
    end = time.time()
    print("Duration:", end - start)
