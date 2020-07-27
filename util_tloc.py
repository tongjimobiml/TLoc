
# coding: utf-8

# In[1]:

from numpy import *
import numpy as np
import pandas as pd
import math as Math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

lonStep_1m = 0.0000105
latStep_1m = 0.0000090201
# rcParams['figure.figsize'] = 12, 14

class RoadGrid:
    def __init__(self, label, grid_size):
        length = grid_size*latStep_1m
        width = grid_size*lonStep_1m
        self.length = length
        self.width = width
        def orginal_plot(label):
            tr = np.max(label,axis=0)
            tr[0]+=25*lonStep_1m
            tr[1]+=25*latStep_1m
            # plot(label[:,0], label[:,1], 'b,')
            bl = np.min(label,axis=0)
            bl[0]-=25*lonStep_1m
            bl[1]-=25*latStep_1m

            # width = (tr[1]-bl[1])/100
            # wnum =int(np.ceil((tr[1]-bl[1])/length))
            # for j in range(wnum):
                # hlines(y = bl[1]+length*j, xmin = bl[0], xmax = tr[0], color = 'red')

            # lnum = int(np.ceil((tr[0]-bl[0])/width))
            # for j in range(lnum):
                # vlines(x = bl[0]+width*j, ymin = bl[1], ymax = tr[1], color = 'red')
            return bl[0], tr[0], bl[1], tr[1]



        xl,xr,yb,yt = orginal_plot(label)
        self.xl = xl
        self.xr = xr
        self.yb = yb
        self.yt = yt
        gridSet = set()
        grid_dict = {}
        for pos in label:
            lon = pos[0]
            lat = pos[1]

            m = int((lon-xl)/width)
            n = int((lat-yb)/length)
            if (m,n) not in grid_dict:
                grid_dict[(m,n)] = []
            grid_dict[(m,n)].append((lon, lat))
            gridSet.add((m,n))
        # print len(gridSet)
        gridlist = list(gridSet)

        grid_center = [tuple(np.mean(np.array(lonlat_list),axis=0)) for (i,j), lonlat_list in grid_dict.items()]


        # for gs in gridSet:
            # xlon = xl+gs[0]*width
            # ylat = yb+gs[1]*length
            # bar(xlon,length,width,ylat,color='#7ED321')
        self.gridlist = gridlist

        self.grids = [(xl+i[0]*width,yb + i[1]*length) for i in grid_dict.keys()]
        self.grid_center=grid_center
        self.n_grid = len(self.grid_center)

    def transform(self, label, sparse=True):
        def one_hot(idx, n):
            a = [0] * n
            a[idx] = 1
            return a
        grid_pos = [self.gridlist.index((int((i[0]-self.xl)/self.width),int((i[1]-self.yb)/self.length))) for i in label]
        if sparse:
            grid_pos = np.array([one_hot(x, len(self.gridlist)) for x in grid_pos], dtype=np.int32)
        return grid_pos

class AreaGrid:
    def __init__(self, k, min_lon, min_lat, max_lon, max_lat):
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.max_lon = max_lon
        self.max_lat = max_lat
        self.k = k
        self.grids = []
        # lonStep_1m = 0.0000105
        # latStep_1m = 0.0000090201
        self.lon_offset = (self.max_lon - self.min_lon) / self.k
        self.lat_offset = (self.max_lat - self.min_lat) / self.k
        for i in xrange(self.k):
            for j in xrange(self.k):
                id = i*self.k + j + 1
                lon = self.min_lon + j * self.lon_offset
                lat = self.min_lat + i * self.lat_offset
                grid = (id, ((lon, lat), (lon+self.lon_offset, lat+self.lat_offset)))
                self.grids.append(grid)

    def gid(self, pt):
        lon, lat = pt
        if self.min_lon <= lon <= self.max_lon and \
                self.min_lat <= lat <= self.max_lat:
            i = int((lat - self.min_lat) / self.lat_offset)
            j = int((lon - self.min_lon) / self.lon_offset)
            return i*self.k+j+1
        else:
            return -1

    def within(self, pt, grid):
        lon, lat = pt
        (ld_lon, ld_lat), (ru_lon, ru_lat) = grid
        return True if (ld_lon <= lon <= ru_lon) and (ld_lat <= lat <= ru_lat) else False

    def dump_boundary(self, filename):
        f_out = open(filename, 'w')
        for id, ((x1, y1), (x2, y2)) in self.grids:
            f_out.write('%.6f,%.6f,%.6f,%.6f\n' % (x1, y1, x2, y2))
        f_out.close()



def calcMean(x,y):
    sum_x = sum(x)
    sum_y = sum(y)
    n = len(x)
    x_mean = float(sum_x+0.0)/n
    y_mean = float(sum_y+0.0)/n
    return x_mean,y_mean

def calcPearson(x,y):
    x_mean,y_mean = calcMean(x,y)	#计算x,y向量平均值
    n = len(x)
    sumTop = 0.0
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(n):
        sumTop += (x[i]-x_mean)*(y[i]-y_mean)
    for i in range(n):
        x_pow += math.pow(x[i]-x_mean,2)
    for i in range(n):
        y_pow += math.pow(y[i]-y_mean,2)
    sumBottom = math.sqrt(x_pow*y_pow)
    p = sumTop/sumBottom
    return p


# In[3]:

from scipy.spatial.distance import pdist
def distribution(data,low=-110,up=-50,r=5,bins=13):
    p_list=np.zeros((bins))
    i=0
    total=len(data)
    while low + r <= up:
        for t in data:
            if t >= low and t < low + r:
                p_list[i] += 1
        low += r
        i += 1
    p_u_list=list()
    if len(data)>0:
        for t in p_list:
            p_u_list.append(float(t)/float(total))
    else:
        p_u_list=p_list.tolist()
    return p_u_list
    
def p_norm_distance(x,y):
    #print x,y
    X = np.vstack([x,y])
    d2 = pdist(X,'minkowski',p=3)
    return d2


# In[4]:

rc = 6378137
rj = 6356725
from math import atan, cos, asin, sqrt, pow, pi, sin
def rad(d):
    return d * math.pi / 180.0

def azimuth(pt_a, pt_b):
    lon_a, lat_a = pt_a
    lon_b, lat_b = pt_b
    rlon_a, rlat_a = rad(lon_a), rad(lat_a)
    rlon_b, rlat_b = rad(lon_b), rad(lat_b)
    ec=rj+(rc-rj)*(90.-lat_a)/90.
    ed=ec*cos(rlat_a)

    dx = (rlon_b - rlon_a) * ec
    dy = (rlat_b - rlat_a) * ed
    if dy == 0:
        angle = 90. 
    else:
        angle = atan(abs(dx / dy)) * 180.0 / pi
    dlon = lon_b - lon_a
    dlat = lat_b - lat_a
    if dlon > 0 and dlat <= 0:
        angle = (90. - angle) + 90
    elif dlon <= 0 and dlat < 0:
        angle = angle + 180 
    elif dlon < 0 and dlat >= 0:
        angle = (90. - angle) + 270 
    return angle

def distance(true_pt, pred_pt):
    lat1 = float(true_pt[1])
    lng1 = float(true_pt[0])
    lat2 = float(pred_pt[1])
    lng2 = float(pred_pt[0])
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * Math.asin(Math.sqrt(Math.pow(Math.sin(a/2),2) +
    Math.cos(radLat1)*Math.cos(radLat2)*Math.pow(Math.sin(b/2),2)))
    s = s * 6378.137
    s = round(s * 10000) / 10
    return s


# In[5]:

col_name_new = [
    #'MRTime',
    'RNCID_1',
    'CellID_1',
    'AsuLevel_1',
    'Dbm_1',
    'SignalLevel_1',
    'RNCID_2',
    'CellID_2',
    'AsuLevel_2',
    'Dbm_2',
    'SignalLevel_2',
    'RNCID_3',
    'CellID_3',
    'AsuLevel_3',
    'Dbm_3',
    'SignalLevel_3',
    'RNCID_4',
    'CellID_4',
    'AsuLevel_4',
    'Dbm_4',
    'SignalLevel_4',
    'RNCID_5',
    'CellID_5',
    'AsuLevel_5',
    'Dbm_5',
    'SignalLevel_5',
    'RNCID_6',
    'CellID_6',
    'AsuLevel_6',
    'Dbm_6',
    'SignalLevel_6',
    'RNCID_7',
    'CellID_7',
    'AsuLevel_7',
    'Dbm_7',
    'SignalLevel_7',
    #'RSSI_6',
]


# In[20]:

def merge_2g_engpara():
    eng_para = pd.read_csv('2g/BS_ALL.csv', encoding='gbk')
    eng_para = eng_para[['RNCID_1', 'CellID_1', 'Lon','Lat']]
    #eng_para = eng_para[eng_para.LAC.notnull() & eng_para[u'经度'].notnull()]
    eng_para = eng_para.drop_duplicates()
    #eng_para.rename(columns={u'经度': 'lon', u'纬度': 'lat'}, inplace=True)
    return eng_para


def make_rf_dataset(data, eng_para):
    for i in range(1, 8):
        data = data.merge(eng_para, left_on=['RNCID_%d' % i, 'CellID_%d' % i], right_on=['RNCID_1','CellID_1'], how='left', suffixes=('', '%d' % i))
        temp=data['CellID_%d'% i].tolist()
        new = list()
        for item in temp:
            if math.isnan(item):
                new.append(0)
            elif int(item)<=0:
                new.append(0)
            else:
                new.append(item)
        data['CellID_%d' % i]=new
    data = data.fillna(-999.)
    #print data.columns
    
    feature = data[col_name_new+['MRTime','TrajID','Lon','Lat','Lon2','Lat2','Lon3','Lat3','Lon4','Lat4',
                                'Lon5','Lat5','Lon6','Lat6','Longitude', 'Latitude']]
    feature['re_lon'] = feature['Longitude'] - feature['Lon']
    feature['re_lat'] = feature['Latitude'] - feature['Lat']
    
    rg = RoadGrid(feature[['re_lon', 're_lat']].values, 20)
    feature['re_ID'] = rg.transform(feature[['re_lon', 're_lat']].values, False)
   
    label = data[['Longitude', 'Latitude']]

    return feature, label, rg

#eng_para = merge_2g_engpara()
eng_para = merge_2g_engpara()

non_tran_f =['RNCID_1',
    'CellID_1',
    'AsuLevel_1',
    'Dbm_1',
    'SignalLevel_1',
    'RNCID_2',
    'CellID_2',
    'AsuLevel_2',
    'Dbm_2',
    'SignalLevel_2',
    'RNCID_3',
    'CellID_3',
    'AsuLevel_3',
    'Dbm_3',
    'SignalLevel_3',
    'RNCID_4',
    'CellID_4',
    'AsuLevel_4',
    'Dbm_4',
    'SignalLevel_4',
    'RNCID_5',
    'CellID_5',
    'AsuLevel_5',
    'Dbm_5',
    'SignalLevel_5',
    'RNCID_6',
    'CellID_6',
    'AsuLevel_6',
    'Dbm_6',
    'SignalLevel_6',
    'Lon','Lat','Lon2','Lat2','Lon3','Lat3','Lon4','Lat4','Lon5','Lat5','Lon6']


import time
import datetime

def compute_time_interval(start, end):
    start = datetime.datetime.fromtimestamp(start / 1000.0)
    end = datetime.datetime.fromtimestamp(end / 1000.0)

    seconds = (end- start).seconds
    
    return seconds



def generate_bs_relative_list(domain, idx):
    domain_f = domain[domain['Lon%d' % idx] > 0]
    rpos = domain_f[['Lon%d' % idx, 'Lat%d' % idx]].values
    serving_bs_pos = domain[['Lon','Lat']].iloc[0,:]
    re_bs_pos = np.zeros((rpos.shape[0], 2))
    for l_idx, val in enumerate(rpos):
        re_bs_pos[l_idx, 0] = val[0] - serving_bs_pos[0]
        re_bs_pos[l_idx, 1] = val[1] - serving_bs_pos[1]
    return re_bs_pos


def generate_traj_domain_list(domain):
    trajs = domain.groupby(['TrajID'])
    traj_list = list()
    for n, traj in trajs:
        traj = traj.sort('MRTime')
        re_traj_pos = np.zeros((traj.shape[0], 2))
        serving_bs_pos = domain[['Lon','Lat']].iloc[0,:]
        for l_idx, val in enumerate(traj[['Longitude','Latitude']].values):
            re_traj_pos[l_idx, 0] = val[0] - serving_bs_pos[0]
            re_traj_pos[l_idx, 1] = val[1] - serving_bs_pos[1]
        traj_list.append(re_traj_pos)
    
    return traj_list



def bs_pair_dis_between_domains(re_bs_pos_1, re_bs_pos_2):
    num_bs_1 = re_bs_pos_1.shape[0]
    num_bs_2 = re_bs_pos_2.shape[1]
    
    dis_sum = 0
    
    if num_bs_1 != 0 and num_bs_2 != 0:
        for pos1 in re_bs_pos_1:
            for pos2 in re_bs_pos_2:
                dis_sum += distance(pos1, pos2)

        return dis_sum / (num_bs_1*num_bs_2)
    else:
        return 999




def mr_rssi_dis(domain_1, domain_2, rss_domain, bs_relative_pos_domain):
    rssi_dis_list = np.zeros((6))
    bs_pair_dis_list = np.zeros((6))
    
    for i in range(1, 7):
        rssi_dis_list[i - 1] = p_norm_distance(rss_domain[domain_1, i], rss_domain[domain_2, i])
        if i > 1:
            bs_pair_dis_list[i -1] = bs_pair_dis_between_domains(bs_relative_pos_domain[domain_1, i], 
                                                                 bs_relative_pos_domain[domain_2, i])
    exp_sum_dis = 0
    for i in range(1, 7):
        exp_sum_dis += math.exp(-bs_pair_dis_list[i-1])
    weight_list = list()
    for i in range(1, 7):
        weight_list.append(math.exp(-bs_pair_dis_list[i-1])/ exp_sum_dis)
    
    total_mr_rssi_dis = 0
    for i in range(1, 7):
        total_mr_rssi_dis += weight_list[i-1]*rssi_dis_list[i-1]
    
    return total_mr_rssi_dis




def calculate_frechet_distance(dp, i, j ,curve_a, curve_b):
    if dp[i][j] > -1:
        return dp[i][j]
    elif i == 0 and j ==0:
        dp[i][j] = distance(curve_a[0], curve_b[0])
    elif i > 0 and j == 0:
        dp[i][j] = max(calculate_frechet_distance(dp, i - 1, 0, curve_a, curve_b), distance(curve_a[i], curve_b[0]))
    elif i == 0 and j > 0:
        dp[i][j] = max(calculate_frechet_distance(dp, 0, j - 1, curve_a, curve_b), distance(curve_a[0], curve_b[j]))
    elif i > 0 and j > 0:
        dp[i][j] = max(min(calculate_frechet_distance(dp, i - 1, j, curve_a, curve_b), calculate_frechet_distance(dp, i - 1, j - 1, curve_a, curve_b), 
                           calculate_frechet_distance(dp, i, j - 1, curve_a, curve_b)), distance(curve_a[i], curve_b[j]))
    else:
        dp[i][j] = float("inf")
    return dp[i][j]


# In[43]:

def get_traj_similarity(curve_a, curve_b):
    dp = [[-1 for _ in range(len(curve_b))] for _ in range(len(curve_a))]
    similarity =  calculate_frechet_distance(dp, len(curve_a)-1, len(curve_b)-1, curve_a, curve_b)
    return max(np.array(dp).reshape(-1, 1))[0]


# In[44]:

def domain_traj_simi(domain_1, domain_2, traj_domain):
    traj_num_1 = len(traj_domain[domain_1])
    traj_num_2 = len(traj_domain[domain_2])
    dis_sum = 0
    
    for trj1 in traj_domain[domain_1]:
        for trj2 in traj_domain[domain_2]:
            dis_sum += get_traj_similarity(trj1, trj2)

    return dis_sum / (traj_num_1*traj_num_1)



def topk_query(dis_arrary, k, domain_name):
    temp = sorted(dis_arrary)
    idx_list, source_domain_name = list(), list()
    for i in range(1, k+1):
        idx_list.append(dis_arrary.index(temp[i]))
    
    for idx in idx_list:
        source_domain_name.append(domain_name[idx])
    return source_domain_name


# In[51]:

def compute_relative_feature(data):
    loc_bs = data[['Lon', 'Lat']].values
    loc_bs = loc_bs[0]
    
    def compute_relative_ID_x(re_x):
        d = distance([re_x, 0], [0, 0])
        if re_x < 0:
            return -abs(int(d / 50))
        else:
            return abs(int(d / 50))
    
    def compute_relative_ID_y(re_y):
        d = distance([0, re_y], [0, 0])
        if re_y < 0:
            return -(int(d / 50))
        else:
            return int(d / 50)
        
    for i in range(2,7):
        data['Lon%d_re' % i] = data['Lon%d' % i] - loc_bs[0]
        data['Lat%d_re' % i] = data['Lat%d' % i] - loc_bs[1]
    
    for i in range(2,7):
        data['re_ID_x_%d' % i] = data.apply(lambda r: compute_relative_ID_x(r['Lon%d_re' % i]), axis = 1)
        data['re_ID_y_%d' % i] = data.apply(lambda r: compute_relative_ID_y(r['Lat%d_re' % i]), axis = 1)
   
    column=['Lon2_re','Lat2_re','Lon3_re','Lat3_re','Lon4_re','Lat4_re','Lon5_re','Lat5_re','Lon6_re','Lat6_re',
           'AsuLevel_1','Dbm_1','SignalLevel_1','AsuLevel_2','Dbm_2','SignalLevel_2','AsuLevel_3','Dbm_3','SignalLevel_3',
           'AsuLevel_4','Dbm_4','SignalLevel_4','AsuLevel_5','Dbm_5','SignalLevel_5','AsuLevel_6','Dbm_6','SignalLevel_6',
           're_ID_x_2', 're_ID_x_3','re_ID_x_4','re_ID_x_5','re_ID_x_6',
           're_ID_y_2', 're_ID_y_3','re_ID_y_4','re_ID_y_5','re_ID_y_6',]
    
    data_re = data[column]
    return data_re


# In[54]:

def non_transfer_train_on_each_domain(domain_tr, domain_te):
    serving_bs = domain_tr[['Lon','Lat']].iloc[0,:]
   
    tr_label = domain_tr[['re_lon', 're_lat']].values
    te_label = domain_te[['re_lon', 're_lat']].values
    tr_f = compute_relative_feature(domain_tr)
    te_f = compute_relative_feature(domain_te)
    
    est = RandomForestRegressor( n_jobs=-1,
    n_estimators = 100,
    max_features='sqrt').fit(tr_f.values, tr_label)
    
    pred = est.predict(te_f.values)
    pred[:, 0] += serving_bs[0]
    pred[:, 1] += serving_bs[1]
    
    error = [distance(pt1, pt2) for pt1, pt2 in zip(pred, domain_te[['Longitude','Latitude']].values)]
    error = sorted(error)
    #print np.median(error)
    return np.median(error), error


# In[55]:

def perpare_source_df(tr_feature_r, source_list):
    source_num = len(source_list)
    source_df = pd.DataFrame()
    source_l = []
    for idx, s_n in enumerate(source_list):
        source = tr_feature_r[(tr_feature_r['RNCID_1']==s_n[0]) & (tr_feature_r['CellID_1']==s_n[1])]
        source_label = source['re_ID'].values
        source_re = compute_relative_feature(source)
        
        if idx == 0:
            source_df = source_re
            source_l = source_label
        else:
            source_df = source_df.append(source_re)
            source_l = np.hstack((source_l, source_label))
    return source_df, source_l


import TRF as STL
reload(STL)


def struct_transfer(domain_tr, domain_te, source_df, source_l, rg):
    serving_bs = domain_tr[['Lon','Lat']].iloc[0,:]
    
    tr_label = domain_tr['re_ID'].values
    #te_label = domain_te[['re_lon', 're_lat']].values
    tr_f = compute_relative_feature(domain_tr)
    te_f = compute_relative_feature(domain_te)
   
    #print source_l, tr_label
    STRUT_RF, C_l = STL.STRUT(source_df.values, source_l, tr_f.values, tr_label, n_trees=50, verbos = False)
    pred = np.asarray(map(lambda x:STL.forest_predict_ensemble(STRUT_RF, x, C_l), te_f.values))
    #pred = np.array([label_all[idx] for idx in pred])
    
    pred_loc = np.array([rg.grid_center[idx] for idx in pred])
    #pred_loc = np.array(pred_loc)
    pred_loc[:, 0] += serving_bs[0]
    pred_loc[:, 1] += serving_bs[1]
    error = [distance(pt1, pt2) for pt1, pt2 in zip(pred_loc, domain_te[['Longitude','Latitude']].values)]
    error = sorted(error)
    #print np.median(error)
    return np.median(error), error

