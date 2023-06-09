#Method Description:
#
#In general, I used a hybrid recommender system, and the hybrid type is switching.
#The dominant component is model-based, in which I added many features from provided datasets into the XGBoost Regression model.
#Review count, number of friends, average star, integrated characteristics such as userful/funny/cool/fans, compliment-related #characteristics and elite are extracted for each user from user.json.
#Review count, average star, normalized longitude and latitude, is open and number of categories are extracted for each business #from business.json.
#Number of photos is extracted for each business from photo.json.
#Number of review likes is extracted for each (user,business) pair from tip.json.
#Number of timestamps is extracted for each business from checkin.json.
#The parameters in XGBoost Regression model such as max_depth, n_estimators and reg_lambda are carefully selected.
#When the predicted situation is not ideal or the regression model is not appropriate for the current predicted pair, we switched to 
#the item-based Collaborative filtering recommendation system and computed the prediction as the final result.
#
#
#Error Distribution:
#>=0 and <1:  101917                                                                                                                 #>=1 and <2:  34762                                                                                                                  #>=2 and <3:  4296                                                                                                                    #>=3 and <4:  1067                                                                                                                    #>=4:  2
#
#
#RMSE:
#0.9573731824016083
#
#
#Execution Time:
#376.90631318092346s


import math
from pyspark import SparkContext
import time
import sys
import json
from operator import add
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error


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
    features = []
    u_feat = user_dict.get(u,[])
    b_feat = business_dict.get(b,[])
    if len(u_feat) == 6:
        for i in range(6):
            features.append(user_dict[u][i])
    else:
        for j in range(6):
            features.append(np.nan)
    if len(b_feat) == 6:
        for i in range(6):
            features.append(business_dict[b][i])
    else:
        for j in range(6):
            features.append(np.nan)
    features.append(photo_dict.get(b,np.nan))
    features.append(tip_dict.get((u,b),np.nan))
    features.append(checkin_dict.get(b,np.nan))
    return features


if __name__ == '__main__':
    start = time.time()
    # execution format: /opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit competition.py <folder_path> <test_file_name> <output_file_name>
    folder = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]
    train_file = folder+"yelp_train.csv"
    user_file = folder+"user.json"
    business_file = folder+"business.json"
    photo_file = folder+"photo.json"
    checkin_file = folder+"checkin.json"
    tip_file = folder+"tip.json"

    sc = SparkContext.getOrCreate()
    sc.setLogLevel('Error')

    RDD0 = sc.textFile(train_file).map(lambda x: x.split(",")).filter(lambda x: x[0] != 'user_id')  # remove the header line in csv
    train_set = RDD0.map(lambda x: ((x[0], x[1]), float(x[2]))).cache()  # {(user_id,business_id),star}
    RDD1 = sc.textFile(test_file).map(lambda x: x.split(",")).filter(lambda x: x[0] != 'user_id')
    val_set = RDD1.map(lambda x: ((x[0], x[1]), float(x[2]))).cache()  # {(user_id,business_id),star}
    val_pair = RDD1.map(lambda x: (x[0], x[1])).cache() # (user_id,business_id)

    # item_based
    item_avg = RDD0.map(lambda x: (x[1], float(x[2]))).groupByKey().map(lambda y: (y[0], sum(y[1]) / len(y[1])))  # (business_id, average_star)
    user_info2 = RDD0.map(lambda x: (x[0], x[1])).groupByKey().map(lambda y: (y[0], list(y[1])))  # (user_id,[business_ids])
    business_info2 = RDD0.map(lambda x: (x[1], x[0])).groupByKey().map(lambda y: (y[0], list(y[1])))  # (business_id,[user_ids])
    user_dict2 = user_info2.collectAsMap()
    business_dict2 = business_info2.collectAsMap()
    rate_dict = train_set.collectAsMap()
    itemavg_dict = item_avg.collectAsMap()
    result1 = RDD1.map(lambda x: predict(x[0], x[1])).collect()


    user_info = sc.textFile(user_file).map(lambda x: json.loads(x))
    user_feature = user_info.map(lambda x: (x["user_id"], (int(x["review_count"]), int(len(x["friends"].split(",")) if x["friends"] is not None else 0), float(
        x["average_stars"]), int(x['useful']) + int(x['funny']) + int(x['cool']) + int(x['fans']), int(x['compliment_hot']) + int(x['compliment_more']) + int(x['compliment_profile']) + int(x['compliment_cute']) + int(
    x['compliment_list']) + int(x['compliment_note']) + int(x['compliment_plain']) + int(x['compliment_cool']) + int(
    x['compliment_funny']) + int(x['compliment_writer']) + int(x['compliment_photos']), int(len(x["elite"]) if x["elite"] is not None else 0))))
    # user features: [review count, number of friends, average star, integrated characteristics, compliment-related characteristics, elite]
    business_info = sc.textFile(business_file).map(lambda x: json.loads(x))
    business_feature = business_info.map(lambda x: (x["business_id"], (int(x["review_count"]), float(x["stars"]), (float(x["longitude"])+180)/360 if x["longitude"] is not None else 0.5, (float(x["latitude"])+90)/180 if x["latitude"] is not None else 0.5, int(x["is_open"]), int(len(x["categories"].split(",")) if x["categories"] is not None else 0))))
    # business features: [review count, average star, normalized longitude, normalized latitude, is open, number of categories]
    photo_info = sc.textFile(photo_file).map(lambda x: json.loads(x))
    photo_feature = photo_info.map(lambda x: (x["business_id"], 1)).reduceByKey(add) # photo feature: number of photos for a business
    tip_info = sc.textFile(tip_file).map(lambda x: json.loads(x))
    tip_feature = tip_info.map(lambda x: ((x["user_id"], x["business_id"]), x["likes"])).reduceByKey(add) # tip feature: number of review likes for a (user,business) pair
    checkin_info = sc.textFile(checkin_file).map(lambda x: json.loads(x))
    checkin_feature = checkin_info.map(lambda x: (x["business_id"], len(x["time"]))).reduceByKey(add) # checkin feature: number of timestamp keys for a business

    user_dict = user_feature.collectAsMap()
    business_dict = business_feature.collectAsMap()
    photo_dict = photo_feature.collectAsMap()
    tip_dict = tip_feature.collectAsMap()
    checkin_dict = checkin_feature.collectAsMap()

    x_train = train_set.map(lambda x: get_feature(x[0][0], x[0][1])).collect()
    y_train = train_set.map(lambda x: x[1]).collect()
    x_val = val_set.map(lambda x: get_feature(x[0][0], x[0][1])).collect()
    y_test = val_set.map(lambda x: x[1]).collect()
    xgb_model = xgb.XGBRegressor(objective='reg:linear', learning_rate=0.1, max_depth=5, n_estimators=300, reg_lambda=1.5, n_jobs=-1)
    xgb_model.fit(x_train, y_train)
    y_pred = xgb_model.predict(x_val)
    differences0 = np.absolute(y_pred - y_test)
    final_pred = []
    for k in range(len(differences0)):
        if differences0[k]<=2:
            final_pred.append(float(y_pred[k]))
        else:
            final_pred.append(float(result1[k][1]))
    print("RMSE:", mean_squared_error(y_test, final_pred))
    differences = np.absolute(np.array(final_pred) - y_test)
    c1, c2, c3, c4, c5 = 0, 0, 0, 0, 0
    for i in range(len(differences)):
        if differences[i] < 1 and differences[i] >= 0:
            c1 += 1
        elif differences[i] < 2 and differences[i] >= 1:
            c2 += 1
        elif differences[i] < 3 and differences[i] >= 2:
            c3 += 1
        elif differences[i] < 4 and differences[i] >= 3:
            c4 += 1
        else:
            c5 += 1
    print("Error Distribution:")
    print(">=0 and <1: ",c1)
    print(">=1 and <2: ",c2)
    print(">=2 and <3: ",c3)
    print(">=3 and <4: ",c4)
    print(">=4: ",c5)
    result = list(zip(val_pair.collect(), final_pred))  # [((user_id,business_id),pred_star)]
    with open(output_file, 'w+') as f:
        f.write("user_id, business_id, prediction\n")
        for row in result:
            f.write(row[0][0] + ',' + row[0][1] + ',' + str(row[1]) + '\n')
        f.close()
    end = time.time()
    print("Duration:", end - start)