import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

def mae_r(y_test, y_pred):
    sbp_mae = round(mean_absolute_error(y_test[:,0], y_pred[:,0]),2)
    dbp_mae = round(mean_absolute_error(y_test[:,1], y_pred[:,1]),2)
    sbp_coef = round(pearsonr(y_test[:, 0], y_pred[:,0])[0], 2)
    dbp_coef = round(pearsonr(y_test[:, 1], y_pred[:,1])[0], 2)
    return sbp_mae, sbp_coef, dbp_mae, dbp_coef

def texts(save_path, attempt_cnt, x_train, x_val, sbp_mae, sbp_coef, dbp_mae, dbp_coef):
    with open(save_path+"/result_"+str(attempt_cnt)+".txt", 'w') as file:
        file.write("Used Data number for train: " + str(x_train.shape[0]) + ", validation: "+ str(x_val.shape[0]) + "\n")
        file.write("MAE of SBP: " +str(sbp_mae)+"\n")
        file.write("MAE of DBP: " +str(dbp_mae)+"\n")
        file.write("Pearson-R of SBP: " +str(sbp_coef)+"\n")
        file.write("Pearson-R of DBP: " +str(dbp_coef)+"\n")
    file.close()
    return