
# coding: utf-8

# In[ ]:

from numpy import *
import numpy as np
import pandas as pd
import math as Math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import util_tloc as utl
reload(utl)


# In[ ]:

data_2g=pd.read_csv("2g/data_2g.csv")
data_2g = data_2g.drop_duplicates(utl.col_name_new)
train, label, rg = utl.make_rf_dataset(data_2g, utl.eng_para)
data_2g = train
from sklearn.cross_validation import train_test_split
tr_feature_r, te_feature_r, tr_label_, te_label_ = train_test_split(train, label, test_size=0.2, random_state=30)


# 按主基站进行分组，每一组为一个domain


domains = tr_feature_r.groupby(['RNCID_1', 'CellID_1'])


# In[ ]:

rss_domain, bs_relative_pos_domain, traj_domain = dict(), dict(), dict()


# In[ ]:

domain_name = list()


#计算每个domain内训练数据的信号强度，基站列表，GPS轨迹

# In[ ]:

for name, domain in domains:
    domain_name.append(name)
    for i in range(1, 7):
        rss_domain[name, i] = utl.distribution(domain['Dbm_%d' % i])
        if i > 1:
            bs_relative_pos_domain[name, i] = utl.generate_bs_relative_list(domain, i)
    traj_domain[name] = utl.generate_traj_domain_list(domain)
            


#计算domain之间的MR distance

# In[ ]:

mr_dis_mat = np.zeros((len(domain_name), len(domain_name)))


# In[ ]:

for d_idx, val in enumerate(domain_name):
    for d_idx_sub, val_sub in enumerate(domain_name):
        if d_idx != d_idx_sub:
            mr_dis_mat[d_idx, d_idx_sub] = mr_dis_mat[d_idx_sub, d_idx] = utl.mr_rssi_dis(val, val_sub, 


# In[ ]:

mr_dis_mat


#计算domain之间的Position distance

# In[ ]:

trj_dis_mat = np.zeros((len(domain_name), len(domain_name)))


# In[ ]:

for d_idx, val in enumerate(domain_name):
    for d_idx_sub, val_sub in enumerate(domain_name):
        if d_idx != d_idx_sub:
            trj_dis_mat[d_idx, d_idx_sub] = utl.domain_traj_simi(val, val_sub, traj_domain)


# In[ ]:

trj_dis_mat


#计算最终的domain distance

# In[ ]:

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
trj_dis_mat_scale = min_max_scaler.fit_transform(trj_dis_mat)
min_max_scaler1 = preprocessing.MinMaxScaler()
mr_dis_mat_scale = min_max_scaler1.fit_transform(mr_dis_mat)

domain_dis_mat = 0.5*mr_dis_mat_scale+ 0.5*trj_dis_mat_scale


#对每个domain内的测试数据进行定位

# In[ ]:

error_tran = []
for name, domain in domains:
    domain_te = te_feature_r[(te_feature_r['RNCID_1'] == name[0]) & (te_feature_r['CellID_1'] == name[1])]
    if domain_te.shape[0] >= 5:
        non_error, raw_error_list = utl.non_transfer_train_on_each_domain(domain, domain_te)
        
        #当non-transfer的中位误差大于30米时，需要进行迁移学习
        if non_error > 30:
            domain_idx = domain_name.index(name)
            # search source domains
            source_list = utl.topk_query(list(domain_dis_mat[domain_idx]), 3, domain_name)
            source_df, source_l = utl.perpare_source_df(tr_feature_r, source_list)
            # structure transfer for random forest
            trans_err, trans_err_list = utl.struct_transfer(domain, domain_te, source_df, source_l, rg)
            if len(list(error_tran)) == 0:
                error_tran = trans_err_list
            else:
                error_tran = np.hstack((error_tran, trans_err_list))
        else:
            if len(list(error_tran)) == 0:
                error_tran = raw_error_list
            else:
                error_tran = np.hstack((error_tran, raw_error_list))


# In[ ]:

error_tran = sorted(error_tran)


# In[ ]:

print "After transfer mean error:", np.mean(error_tran), "median error:", np.median(error_tran)


# In[ ]:

est = RandomForestRegressor(n_jobs=-1, n_estimators = 50, max_features='sqrt').fit(tr_feature_r[utl.non_tran_f], tr_label_)
pred = est.predict(te_feature_r[utl.non_tran_f].values)
    
error = [utl.distance(pt1, pt2) for pt1, pt2 in zip(pred, te_label_.values)]
error = sorted(error)
print "No-transfer mean error:", np.mean(error), "median error:", np.median(error)



