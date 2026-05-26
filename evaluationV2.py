
# from estimators import PF
from estimators import TrackerV2 as ParticleFilterTracker
from application import Settings
import pandas as pd
# from datetime import datetime,timedelta
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from typing import List
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
from tqdm import tqdm
import itertools
import os
import pickle
import concurrent.futures
import threading # Import threading module
import cv2
from glob import glob
from typing import Optional
import warnings
import argparse

# Treat RuntimeWarning as an error
warnings.filterwarnings("error", category=RuntimeWarning)

import multiprocessing
# from cuml.neighbors import KernelDensity as CUDAKernelDensity
# import cupy as cp
os.environ['NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS'] = '0'


# settings = Settings()
advance_window = datetime.timedelta(hours=1)
history_windows = [15,30,45,60,75,90]
history_windows = [datetime.timedelta(minutes=m) for m in history_windows]
dts = [15,30,45,60]
dts = [datetime.timedelta(minutes=m) for m in dts]
min_samples = [3]
max_dists = [.2,.3]
grid_sizes_km = [10,5,2]
steps = [1,2,3,4,5,6]
# steps = [1]
gaussian_density_bandwith = [.1]
# probability_list:List[float] = [.01,.02,.03,.04,.05,.06,.07,.08,.09,.1]
probability_list:List[float] = [.5,.6,.7,.8,.9,.95]
# probability_future_list:List[float] = [.01,.02,.03,.04,.05,.06,.07,.08,.09,.1]
probability_future_list:List[float] = [.99,.95,.9,.8,.7,.6,.5]

results_folder = 'results'
file_to_evaluate = 'ILDN 26-31_01_24.xlsx'
# file_to_evaluate = 'ILDN 2023-2024 season.xlsx'
# file_to_evaluate = 'ENTLN 2022-2023 season.xlsx'
# file_to_evaluate = 'WWLN 2023-2024 season.xlsx'
ground_truth_file = None


results_filename_path = os.path.join(results_folder,f'{file_to_evaluate}_results.pk')

KM_PER_DEGREE = 110.0
MAX_PROCESSES = 6
MAX_THREADS = 150

parameter_combinations = list(itertools.product(
                                            history_windows,
                                            dts,
                                            max_dists,
                                            min_samples,
                                            grid_sizes_km,
                                            gaussian_density_bandwith,
                                            ))
parameter_combinations.sort()

def load_data(filename):
    if filename.endswith('.xlsx'):
        frame = pd.read_excel(filename,index_col=False,sheet_name='Sheet1')
    elif filename.endswith('.csv'):
        frame = pd.read_csv(filename,index_col=False)
        frame['UTC'] = pd.to_datetime(frame['UTC'],format='ISO8601')
    frame = frame.sort_values('UTC',ignore_index=True)
    return frame

def create_mask_from_lon_lat(lon_lat_points, X, Y):
    """
    Creates a binary mask (same shape as X, Y) with 1s at locations
    corresponding to the given lon/lat points, and 0s elsewhere.

    Args:
        lon_lat_points (np.ndarray): A 2D array of shape (N, 2) where N is the
                                     number of points, and each row is [longitude, latitude].
        X (np.ndarray): 2D array of longitudes from meshgrid.
        Y (np.ndarray): 2D array of latitudes from meshgrid.

    Returns:
        np.ndarray: A 2D binary mask (0s and 1s) with the same shape as X and Y.
    """
    Z_new = np.zeros_like(X, dtype=int)

    for target_lon, target_lat in lon_lat_points:
        # Calculate the squared difference for efficiency
        diff_lon = X - target_lon
        diff_lat = Y - target_lat
        distances_sq = diff_lon**2 + diff_lat**2

        # Find the index of the minimum distance
        min_idx_flat = np.argmin(distances_sq)
        row_idx, col_idx = np.unravel_index(min_idx_flat, distances_sq.shape)

        # Set the corresponding cell in the new mask to 1
        Z_new[row_idx, col_idx] = 1
    Z_new = np.expand_dims(Z_new,0)
    return Z_new

def create_prediction_mask(est_array: np.ndarray,
                        #    X:np.ndarray,
                        #    Y:np.ndarray,
                            grid_coords:np.ndarray,
                           x_resolution,
                           y_resolution):
    # grid_coords = np.vstack([X.ravel(), Y.ravel()]).T

    kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
    kde.fit(est_array)
    Z = np.exp(kde.score_samples(grid_coords)).reshape(y_resolution,x_resolution)

    if Z.any() : Z  = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
    Z = np.expand_dims(Z,0)
    return Z

def execute_iteration(frame,parameters,gt_frame=None):
    history_window,dt,max_dist,min_samples,grid_km,gauss_bandwith = parameters
    param_idx = parameter_combinations.index(parameters)
    settings = Settings()

    dict_filename = f'{str(parameters)}.pk'
    dict_path = os.path.join(results_folder,file_to_evaluate,dict_filename)
    if os.path.isfile(dict_path): 
        return

    xminmax = (settings.nswe['w'] - settings.expand, settings.nswe['e'] + settings.expand)
    yminmax = (settings.nswe['s'] - settings.expand, settings.nswe['n'] + settings.expand)
    
    lon_size = settings.nswe['e'] - settings.nswe['w'] + 2 * settings.expand
    lat_size = settings.nswe['n'] - settings.nswe['s'] + 2 * settings.expand
    y_resolution = int(lat_size * KM_PER_DEGREE / grid_km)
    x_resolution = int(lon_size * KM_PER_DEGREE / grid_km)
    
    xlin = np.linspace(xminmax[0], xminmax[1], x_resolution)
    ylin = np.linspace(yminmax[0], yminmax[1], y_resolution)
    X, Y = np.meshgrid(xlin, ylin)
    grid_coords = np.vstack([X.ravel(), Y.ravel()]).T

    steps_data_dict = {step:{'pred':np.zeros((0,*X.shape)),'true':np.zeros((0,*X.shape))} for step in steps}

    region_mask =  ((frame['lon'] >= xminmax[0]) & (frame['lon'] <= xminmax[1]))
    region_mask &= ((frame['lat'] >= yminmax[0]) & (frame['lat'] <= yminmax[1]))
    df = frame[region_mask]

    if not gt_frame is None:
        gt_region_mask =  ((gt_frame['lon'] >= xminmax[0]) & (gt_frame['lon'] <= xminmax[1]))
        gt_region_mask &= ((gt_frame['lat'] >= yminmax[0]) & (gt_frame['lat'] <= yminmax[1]))
        gt_df = gt_frame[gt_region_mask]
        # print(gt_df.head())

    df.loc[:,'UTC'] = pd.to_datetime(df['UTC'])
    # print( type(df['UTC'].iloc[0]) , type(history_window) , type(dt))
    date = df['UTC'].iloc[0] + history_window + dt

    period:datetime.timedelta = df['UTC'].iloc[-1] - date

    n_windows = int(period / advance_window)
    pbar = tqdm(range(n_windows),desc=f'{param_idx}:{str(parameters)}',leave=False,position=os.getpid() % MAX_PROCESSES)
    # pbar = tqdm(range(n_windows),desc=f'Dates thread:{hash(threading.get_ident() % MAX_THREADS)}',leave=False,position=hash(threading.get_ident()) % MAX_THREADS)
    
    pf = ParticleFilterTracker()    
    pf_future = ParticleFilterTracker()
    
    while date <= df['UTC'].iloc[-1]:

        time_mask = (df.UTC >= date - history_window) \
            & (df.UTC <= date)
        time_mask_prev = (df.UTC >= date - dt - history_window) \
            & (df.UTC <= date - dt)

        measurements = df[time_mask][['lon', 'lat']].to_numpy()
        measurements_prev = df[time_mask_prev][['lon', 'lat']].to_numpy()
        
        if len(measurements) <= 0 or len(measurements_prev) <= 0 :
            date += advance_window
            pbar.update(1)
            continue
        # measure_dt = dt.seconds // 60


        pf.init_tracker(max_dist=max_dist,min_samples=min_samples)
        pf.update_all(measurements,measurements_prev,dt)
        particles = pf.get_all_particles()
        if particles.shape[0] <=0: 
            date += advance_window
            pbar.update(1)
            continue
        # for step in tqdm(steps,leave=False,desc="Steps"):
        for step in steps:
            particles = pf.predict(step,dt)
            # try:
            # except Exception as e:
            #     print('-----------------------------------------')
            #     print(e)
            #     print(date)
            #     print('-----------------------------------------')
            # y_pred = create_prediction_mask(particles[:,:2],grid_coords,x_resolution,y_resolution)
            # y_pred = create_prediction_mask_CUDA(particles[:,:2],X,Y,x_resolution,y_resolution)
            # y_pred = create_prediction_mask_CUDA(particles[:,:2],grid_coords,x_resolution,y_resolution,gauss_bandwith)
            y_pred = pf.get_gaussian_estimation(grid_coords=grid_coords,
                                                resolution_x=x_resolution,
                                                resolution_y=y_resolution,
                                                bandwith=gauss_bandwith   )
            y_pred = np.expand_dims(y_pred,0)
            steps_data_dict[step]['pred'] = np.vstack((steps_data_dict[step]['pred'],y_pred))
            

            future_df = df
            if not gt_frame is None:
                future_df = gt_df
            truth_time_mask = (future_df.UTC > date + datetime.timedelta(hours=step-1)) \
                & (future_df.UTC <= date + datetime.timedelta(hours=step))
            future_measurements = future_df[truth_time_mask][['lon', 'lat']].to_numpy()


            # y_true = create_mask_from_lon_lat(future_measurements,X,Y)
            pf_future.init_tracker(max_dist=max_dist,min_samples=min_samples)
            try:
                pf_future.update_all(future_measurements,future_measurements,dt)
            except:
                # print(len(future_measurements))
                pass
            y_true=pf_future.get_gaussian_estimation(grid_coords=grid_coords,
                                                    resolution_x=x_resolution,
                                                    resolution_y=y_resolution,
                                                    bandwith=gauss_bandwith )
            y_true = np.expand_dims(y_true,0)

            steps_data_dict[step]['true'] = np.vstack((steps_data_dict[step]['true'],y_true))

        date += advance_window
        pbar.update(1)
    
    with open(dict_path,'wb') as f:
        pickle.dump(steps_data_dict,f)

    return dict_filename

def execute_parameters_processpool_CUDA(frame,gt_frame=None):
    for i,param in enumerate(parameter_combinations):
        print(f'{i}:{param}')
    # print('# combinations: ',len(parameter_combinations))
    # multiprocessing.set_start_method('spawn', force=True)
    # with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
    #     futures = [executor.submit(execute_iteration,frame,param) for param in parameter_combinations]

    for param in parameter_combinations:
        execute_iteration(frame,param,gt_frame)
        # return

def evaluate():
    
    frame = load_data(file_to_evaluate)
    gt_frame = None
    if ground_truth_file:
        gt_frame = load_data(ground_truth_file)
    # os.makedirs(os.path.join(results_folder,file_to_evaluate,))
    os.makedirs(os.path.join(results_folder,file_to_evaluate),exist_ok=True)
    # execute_parameters_threadpool(frame)
    # execute_parameters_processpool(frame)
    # execute_parameters_processpool_CUDA(frame)
    execute_parameters_processpool_CUDA(frame,gt_frame)


'''
    ############################################ Skill Scores #######################################3
'''

def build_confusion_matrix_manual(y_true, y_pred):
    """
    Builds a confusion matrix from two binary NumPy matrices.

    Args:
        y_true (np.ndarray): The actual (ground truth) binary matrix.
        y_pred (np.ndarray): The predicted binary matrix.

    Returns:
        tuple: A tuple containing (TN, FP, FN, TP)
               Alternatively, you could return a 2x2 NumPy array.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Input matrices must have the same shape.")

    # Flatten the arrays to work with 1D vectors for easier comparison
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Calculate confusion matrix components
    TP = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    TN = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
    FP = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    FN = np.sum((y_true_flat == 1) & (y_pred_flat == 0))

    # A common way to represent the confusion matrix is as a 2x2 array:
    # [[TN, FP],
    #  [FN, TP]]
    confusion_matrix_array = np.array([[TN, FP],
                                       [FN, TP]])

    return confusion_matrix_array

def evaluate_estimation(y_true,y_pred,probability_list:Optional[List[float]]=None):

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    mat_dict = {}
    roc_dict = {}
    prc_dict = {}
    if probability_list is None:
        y_pred_mask = y_pred
        fpr, tpr, thresholds = roc_curve(y_true.astype(int), y_pred_mask)
        # Calculate Youden's J Statistic
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]


        roc_auc = auc(fpr, tpr)

        p = optimal_threshold
        y_pred_mask = np.zeros_like(y_pred,dtype=int)
        y_pred_mask[y_pred >= p] = 1

        # conf_mat = confusion_matrix(y_true.astype(int), y_pred_mask)
        conf_mat = build_confusion_matrix_manual(y_true.astype(int),y_pred_mask)

        mat_dict[p] = conf_mat
        roc_dict[p] = {'fpr':fpr,'tpr':tpr,'roc_auc':roc_auc}


        # prc = precision_recall_curve(y_true.astype(int), y_pred_mask)
        # precision, recall, thresholds_prc = prc

    
    else:
        for p in probability_list:
            # 0.31 s
            y_pred_mask = np.zeros_like(y_pred,dtype=int)
            y_pred_mask[y_pred >= p] = 1
            # 0.30 s

            conf_mat = build_confusion_matrix_manual(y_true.astype(int),y_pred_mask)
            # mat_dict[f'{p}'] = conf_mat
            mat_dict[p] = conf_mat
            # 0.21 s
            fpr, tpr, thresholds = roc_curve(y_true.astype(int), y_pred_mask)
            # 0.0002 s
            roc_auc = auc(fpr, tpr)
            # roc_dict[f'{p}'] = {'fpr':fpr,'tpr':tpr,'roc_auc':roc_auc}
            roc_dict[p] = {'fpr':fpr,'tpr':tpr,'roc_auc':roc_auc}

    return mat_dict,roc_dict

def evaluate_patches_estimation(y_true,y_pred,probability_list):
    
    # A common way to represent the confusion matrix is as a 2x2 array:
    # [[TN, FP],
    #  [FN, TP]]
    conf_dict = {p:np.zeros((2,2)) for p in probability_list} #
    conf_dcit_count = {p:np.zeros((2,2)) for p in probability_list}
    areas_dict = {}

    area_template = np.zeros_like(y_true[0])
    area_template[1:-1,1:-1] = 1
    cnts,_  = cv2.findContours(area_template.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cnt = cnts[0]
    cnt = np.concat((cnt,np.expand_dims(cnt[0],axis=0)),axis=0)
    total_area = cv2.contourArea(cnt)

    for p in probability_list:

        ta,asum = 0,0
        
        
        for true_mat,pred_mat in zip(y_true,y_pred):
            
            Z = np.zeros_like(pred_mat)
            Z[pred_mat >= p] = 1
            T = true_mat

            contours, hierarchy = cv2.findContours(Z.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

            where = np.where(T >= 1)
            points = np.array([where[1],where[0]],dtype=float).T

            set_points = set(map(tuple,points))
            # print(set_points)
            tn = total_area
            
            for cnt in contours:

                cnt = np.concat((cnt,np.expand_dims(cnt[0],axis=0)),axis=0)

                ress = np.array([cv2.pointPolygonTest(cnt,point,False) for point in points]) # 1 or 0 - inside or edge | -1 - outside
                in_points_idx = np.where(ress >= 0)[0]
                set_points -= set(map(tuple,points[in_points_idx]))

                area = int(cv2.contourArea(cnt))
                asum+=area

                count = len(np.where(ress >= 0 )[0])
                is_inside = count > 0 

                # if use_area: 
                    # tp = 0 if len(np.where(ress >= 0 )[0]) <= 0 else area
                # else: 
                    # tp = len(np.where(ress >= 0 )[0]) 
                # fp = 0 if tp > 0 else area
                tn = tn - area

                conf_dict[p][0,1] += area if not is_inside else 0
                conf_dict[p][1,1] += area if is_inside else 0

                conf_dcit_count[p][0,1] += area if not is_inside else 0
                conf_dcit_count[p][1,1] += count if is_inside else 0

            fn = len(set_points)

            conf_dict[p][1,0] += fn
            conf_dict[p][0,0] += tn - fn

            conf_dcit_count[p][1,0] += fn
            conf_dcit_count[p][0,0] += tn - fn

            ta+=total_area
            
        areas_dict[p] = (asum,ta)

    return conf_dict,conf_dcit_count,areas_dict
    
def confusion2skills(tn,fp,fn,tp):
        # Calculate binary classification specific metrics
        total_observations = tp + tn + fp + fn
        accuracy = (tp + tn) / total_observations if total_observations != 0 else 0
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        # specificity = tn / (tn + fp) if (tn + fp) != 0 else 0 # Also known as True Negative Rate
        fall_out = fp / (fp + tn) if (fp + tn) != 0 else 0 # Also known as False Positive Rate
        # negative_predictive_value = tn / (tn + fn) if (tn + fn) != 0 else 0
        # false_discovery_rate = fp / (tp + fp) if (tp + fp) != 0 else 0
        # false_omission_rate = fn / (fn + tn) if (fn + tn) != 0 else 0
        csi = tp / (tp + fn + fp) if (tp + fn + fp) != 0 else 0 # Also known as Critical Success Index (CSI)3
        # bias = (tp + fp) / (tp + fn) if (tp + fn) != 0 else 0 # Ratio of forecasts to observations
        # probability_of_detection = recall # Same as Recall
        # false_alarm_rate = fp / (tp + fp) if (tp + fp) != 0 else 0 # 1 - Precision

        # Heidke Skill Score (HSS) - Version 1
        # Expected accuracy by chance (product of marginals)
        expected_accuracy_numerator = ((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))
        expected_accuracy_hss1 = expected_accuracy_numerator / (total_observations**2) if total_observations != 0 else 0
        hss = (accuracy - expected_accuracy_hss1) / (1 - expected_accuracy_hss1) if (1 - expected_accuracy_hss1) != 0 else 0

        # Heidke Skill Score (HSS) - Version 2 (HSS2)
        # Often simplified to (TP*TN - FP*FN) / ((TP+FN)(FN+TN)+(TP+FP)(FP+TN))
        hss2_numerator = (tp * tn) - (fp * fn)
        hss2_denominator = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))
        hss2 = hss2_numerator / hss2_denominator if hss2_denominator != 0 else 0

        # True Skill Statistic (TSS) / Pierce Skill Score / Hanssen-Kuipers Discriminant
        # TSS = POD - FAR = Recall - Fall-out = (TP / (TP+FN)) - (FP / (FP+TN))
        tss = recall - fall_out

        return accuracy,precision,recall,f1_score,hss,hss2,tss,csi

def paramster2tuple(params_str: str):

    scope = {'datetime': datetime}
    result_tuple = eval(params_str, scope)
    return result_tuple

def maskbyprobability(y_true: np.array,thr):
    Z = y_true.copy()
    dens = Z.reshape(-1)
    sorted_dens = np.sort(dens)[::-1]
    cumulative_sum = np.cumsum(sorted_dens)
    total_sum = np.sum(dens)
    # print(thr,total_sum,cumulative_sum,Z.shape)
    cumulative_prob = cumulative_sum / total_sum
    threshold_index = np.where(cumulative_prob >= thr)[0][0]
    threshold_value = sorted_dens[threshold_index]
    Z[Z < threshold_value] = 0
    Z[Z >= threshold_value] = 1

    return Z

def skillscores2csv(file2eval):

    USE_TQDM = True

    dict_folder_path = os.path.join(results_folder,file2eval)
    dict_files = glob(os.path.join(dict_folder_path,'*pk'))
    # print(len(dict_files))
    # dict_file = dict_files[0]
    # params_str = os.path.basename(dict_file)[:-3]
    # print(dict_file,params_str,paramster2tuple(params_str))

    csv_file = os.path.join(dict_folder_path,'results.csv')
    csv_file_pathces = os.path.join(dict_folder_path,'results_patches.csv')
    csv_file_count = os.path.join(dict_folder_path,'results_count.csv')
    df =         pd.DataFrame(columns=['window_size(minutes)','dt(minutes)','max_dist(deg)','min_samples','grid_km','pred(hours)','thr(%)','thr_future(%)','tn', 'fp', 'fn', 'tp','roc_auc','accuracy','precision','recall','f1','hss','hss2','tss','csi','n_cases','gauss_bandwith'])
    # df =         pd.DataFrame(columns=['window_size(minutes)','dt(minutes)','max_dist(deg)','min_samples','grid_km','pred(hours)','thr(%)','tn', 'fp', 'fn', 'tp','roc_auc','accuracy','precision','recall','f1','hss','hss2','tss','csi','n_cases','gauss_bandwith'])
    df_patches = pd.DataFrame(columns=['window_size(minutes)','dt(minutes)','max_dist(deg)','min_samples','grid_km','pred(hours)','thr(%)','tn', 'fp', 'fn', 'tp','accuracy','precision','recall','f1','hss','hss2','tss','csi','area','total_area','n_cases','gauss_bandwith'])
    df_count =   pd.DataFrame(columns=['window_size(minutes)','dt(minutes)','max_dist(deg)','min_samples','grid_km','pred(hours)','thr(%)','tn', 'fp', 'fn', 'tp','accuracy','precision','recall','f1','hss','hss2','tss','csi','area','total_area','n_cases','gauss_bandwith'])
    
    row_i = 0
    pbar1 = dict_files
    if USE_TQDM:
        pbar1 = tqdm(dict_files, desc='Parameters', position=os.getpid() % MAX_PROCESSES)
    # for dict_file in tqdm(dict_files,desc='Parameters',position=os.getpid() % MAX_PROCESSES):
    for dict_file in pbar1:

        results_dict = pickle.load(open(dict_file,'rb'))
        params_str = os.path.basename(dict_file)[:-3]
        parameters = paramster2tuple(params_str)
        window,dt,max_dist,min_sample,grid_km,gauss_bandwith = parameters
        # print(parameters)
        # continue
        pbar2 = results_dict.items()
        if USE_TQDM:
            pbar2 = tqdm(results_dict.items(),leave=False,desc='Steps',position=(os.getpid()+1) % MAX_PROCESSES)
        # for step,step_dict in tqdm(results_dict.items(),leave=False,desc='Steps',position=(os.getpid()+1) % MAX_PROCESSES):
        for step,step_dict in pbar2:
            y_true = step_dict['true']
            y_pred = step_dict['pred']

            pbar3 = probability_future_list
            if USE_TQDM:
                pbar3 = tqdm(probability_future_list,leave=False,desc='Future probability',position=(os.getpid()+2) % MAX_PROCESSES)
            # for p_future in tqdm(probability_future_list,leave=False,desc='Future probability',position=(os.getpid()+2) % MAX_PROCESSES):
            for p_future in pbar3:

                # y_true_masked = np.zeros_like(y_true)
                # y_true_masked[y_true >= p_future] = 1
                y_true_masked = maskbyprobability(y_true,p_future)
                # print(p_future,np.sum(y_true_masked))

                n_cases = len(y_true)

                mat_dict,roc_dict = evaluate_estimation(y_true_masked,y_pred,probability_list=None)

                # mat_dict,roc_dict = evaluate_estimation(y_true,y_pred,probability_list)
                # mat_dict_patches,conf_dcit_count,areas_dict_patch = evaluate_patches_estimation(y_true,y_pred,probability_list)
                # mat_dict_patches,conf_dcit_count,areas_dict_patch = evaluate_patches_estimation(y_true,y_pred,probability_list)


                for k,p in enumerate(roc_dict.keys()):
                    roc_data = roc_dict[p]
                    conf_mat = mat_dict[p]
                    # print(conf_mat)
                    # conf_mat_patches = mat_dict_patches[p]
                    # conf_mat_count = conf_dcit_count[p]
                    
                    # area_tuple_patch = areas_dict_patch[p]
                    # pred_area,total_area = area_tuple_patch

                    fpr,tpr,roc_auc = roc_data.values()
                    tn, fp, fn, tp = conf_mat.ravel()
                    
                    accuracy,precision,recall,f1_score,hss,hss2,tss,csi = confusion2skills(tn,fp,fn,tp)
                    df_row = [window,dt,max_dist,min_sample,grid_km,step,p,p_future,tn, fp, fn, tp,roc_auc,accuracy,precision,recall,f1_score,hss,hss2,tss,csi,n_cases,gauss_bandwith]
                    df.loc[row_i] = df_row

                    # tn, fp, fn, tp = conf_mat_patches.ravel()
                    # accuracy,precision,recall,f1_score,hss,hss2,tss,csi = confusion2skills(tn,fp,fn,tp)
                    # df_row = [window,dt,max_dist,min_sample,grid_km,step,p,tn, fp, fn, tp,accuracy,precision,recall,f1_score,hss,hss2,tss,csi,pred_area,total_area,n_cases,gauss_bandwith]
                    # df_patches.loc[row_i] = df_row

                    # tn, fp, fn, tp = conf_mat_count.ravel()
                    # accuracy,precision,recall,f1_score,hss,hss2,tss,csi = confusion2skills(tn,fp,fn,tp)
                    # df_row = [window,dt,max_dist,min_sample,grid_km,step,p,tn, fp, fn, tp,accuracy,precision,recall,f1_score,hss,hss2,tss,csi,pred_area,total_area,n_cases,gauss_bandwith]
                    # df_count.loc[row_i] = df_row

                    row_i += 1
        # break
    # print(csv_file_pathces)
    df.to_csv(csv_file)
    # df_patches.to_csv(csv_file_pathces)
    # df_count.to_csv(csv_file_count)


def skillscore_plot():
    pass

def skillscores_runner():

    # list_of_file2eval = ['ILDN 2023-2024 season.xlsx','ENTLN 2022-2023 season.xlsx','WWLN 2023-2024 season.xlsx']#,'ILDN 26-31_01_24.xlsx']
    # list_of_file2eval = ['ILDN 26-31_01_24.xlsx']
    # list_of_file2eval = ['ILDN 2023-2024 season.xlsx']
    # list_of_file2eval = ['WWLN 2023-2024 season.xlsx']
    # list_of_file2eval = ['ENTLN 2022-2023 season.xlsx']
    list_of_file2eval = [file_to_evaluate]
    # multiprocessing.set_start_method('spawn', force=True)
    # with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
    #    futures = [executor.submit(skillscores2csv,file2eval) for file2eval in list_of_file2eval]

    # skillscores2csv(list_of_file2eval[0])
    for file2eval in list_of_file2eval:
        skillscores2csv(file2eval) 

def tests_contours():
    
    results_dict = pickle.load(open(results_filename_path,'rb'))
    for parameters,steps_data_dict in tqdm(results_dict.items(),desc='Parameters'):
        
        # print(parameters)
        # print(steps_data_dict[1]['pred'][0])

        Z = steps_data_dict[6]['pred'][13]
        Z[Z >= .6] = 1
        Z[Z < .6] = 0
        T = steps_data_dict[6]['true'][13]
    
        contours, hierarchy = cv2.findContours(Z.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        # print(np.array([*np.where(T >= 1)]).T)
        where = np.where(T >= 1)
        points = np.array([where[1],where[0]],dtype=float).T
        # coords = np.array()
        # print(list(zip(np.where(T >= 1)[0],np.where(T >= 1)[1])))
        # plt.subplot(1,3,1)
        # cv2.drawContours(Z,contours,-1,(0,255,0),3)
        # cv2.imshow('Z',Z)
        plt.subplot(1,2,1).imshow(Z)
        plt.subplot(1,2,1).grid()
        for coords in contours:
            coords = np.concat((coords,np.expand_dims(coords[0],axis=0)),axis=0)
            plt.subplot(1,2,1).plot(*coords.squeeze(1).T ,marker='o', linestyle='-', color='blue')
            plt.subplot(1,2,2).plot(*coords.squeeze(1).T , linestyle='-', color='blue')

            ress = np.array([cv2.pointPolygonTest(coords,point,False) for point in points])
            area = int(cv2.contourArea(coords))
            tp = 0 if len(np.where(ress >= 0 )[0]) <= 0 else area
            fn = len(np.where(ress < 0)[0])
            fp = 0 if tp > 0 else area
            tn = 0

        print(tp,fn,fp,tn)
        plt.subplot(1,2,2).imshow(T)
        plt.show()

        break

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog='Evaluation v.2.0',
                    description='Run an algorith evalution of a file')
    
    parser.add_argument('-f','--file', type=str, required=True, help="Must provide lightning detection file to evaluate.")           # positional argument
    parser.add_argument('-gt','--gound_truth', type=str, required=False, help="provide lightning Ground Thruth file.")           # positional argument
    args = parser.parse_args()
    file_to_evaluate = args.file
    ground_truth_file = None
    if len(args.gound_truth) > 0: ground_truth_file = args.gound_truth

    # print(args)
    evaluate()
    skillscores_runner()

    # t = datetime.datetime.now()
    # print(datetime.datetime.now() - t)
    
    # print(len(parameter_combinations))
    # tests()

    # a = np.random.randint(0,10,(4,16))
    # print(a.shape,np.sum(a,axis=0).shape,np.sum(a,axis=1).shape)

    pass
