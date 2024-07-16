import os
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import datetime

def time_info(**kwargs):
    current_time = datetime.datetime.now()
    print("Time stamp: ", current_time.strftime("%Y/%m/%d/%H:%M:%S"))
    if "past_time" in kwargs: 
        past_time = kwargs['past_time']
        elapsed_time = current_time - past_time
        print(f"Taken Time: {elapsed_time.seconds//60} min")
    return
    
def gpu_config(gpu_num, gpu_mem): # GPU device & memory config
    # =================GPU_Selection==================
    gpus = tf.config.list_physical_devices('GPU')
    avail_gpu_list = [gpu.name[-1] for gpu in gpus]
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
        print(f"GPU {gpu_num} is selected")
    except:
        print(f"selected {gpu_num} not available.\n only {avail_gpu_list} available")

    # ================GPU_memory_occupancy============
    if gpu_num != "-1":
        if gpu_mem <= 0 or gpu_mem >= 24:
            try:
                for i in range(len(gpus)):
                	tf.config.experimental.set_memory_growth(gpus[i], True)
                print("allowing GPU memory growth. Other GPUs will be set in identical configuration")
            except RuntimeError as e:
                print("GPU memory growth allowance error")
        else:
            try:
                tf.config.set_logical_device_configuration(
                    gpus[int(gpu_num)],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=1024*gpu_mem)])
                print(f"{gpu_mem} GB of memory is set")
            except RuntimeError as e:
                print("GPU memory set error")
    return

def dataset_threshold(y_data, args):
    train_size = args.train_size
    val_size = args.validation_size
    test_size = args.test_size
    req_data_size = train_size + val_size + test_size
    current_data_size = y_data.shape[0]
    if current_data_size >= req_data_size:
        return True
    else:
        print("data size is not big enough to wield the training")
        return False

def dataset_index(data_size, attempt, args):
    train_size = args.train_size
    val_size = args.validation_size
    test_size = args.test_size

    if attempt == 1:
        train_indx = np.random.choice(range(0, train_size), 
                                      size = train_size, replace = False)
        val_indx = np.random.choice(range(train_size, train_size + val_size), 
                                    size = val_size, replace = False)
        test_indx = np.random.choice(range(train_size + val_size, train_size + val_size + test_size), 
                                     size = test_size, replace = False)
    elif attempt == 2:
        train_indx = np.random.choice(range(val_size, val_size + train_size), 
                                      size = train_size, replace = False)
        val_indx = np.random.choice(range(0, val_size), 
                                    size = val_size, replace = False)
        test_indx = np.random.choice(range(train_size + val_size, train_size + val_size + test_size), 
                                     size = test_size, replace = False)
    elif attempt == 3:
        train_indx = np.random.choice(range(test_size + val_size, test_size + val_size + train_size), 
                                      size = train_size, replace = False)
        val_indx = np.random.choice(range(test_size, test_size + val_size), 
                                    size = val_size, replace = False)
        test_indx = np.random.choice(range(0, test_size), 
                                     size = test_size, replace = False)
    return train_indx, val_indx, test_indx

def make_save_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return

def find_files(folder_path, kind): # Search certain type of files and extract the paths as a list
    file_root_list = []
    for (path, dir, files) in os.walk(folder_path):
        for filename in files:
            if filename.endswith(kind):
                file_root = os.path.join(path, filename)
                file_root_list.append(file_root)
    return file_root_list

def add_compile(model, **kwargs): # Compile model
    from tensorflow.keras import optimizers, losses, metrics
    
    opt = optimizers.Adam(learning_rate=kwargs['learning_rate'])
    model.compile(opt, loss='mean_squared_error', metrics = ['mae'])

    return model

# Plot

def BP_dat_plot(dat, title):
    dist_data_sbp = dat[:, 0]
    dist_data_dbp = dat[:, 1]

    sbp_mean = np.mean(dist_data_sbp)
    dist_data_sbp = dist_data_sbp.astype(int)

    dbp_mean = np.mean(dist_data_dbp)
    dist_data_dbp = dist_data_dbp.astype(int)

    sbp_std = np.std(dat[:, 0])
    dbp_std = np.std(dat[:, 1])

    fig_bp_dist, axs = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)

    axs[0].hist(dist_data_sbp, bins=range(min(dist_data_sbp), max(dist_data_sbp)+2))
    axs[0].set_title(title + ' SBP')
    axs[0].axvline(x=sbp_mean, ymin=0, ymax=1, color = 'blue', linestyle = '--')
    axs[0].axvline(x=sbp_mean - sbp_std, ymin=0, ymax=1, color = 'red', linestyle = '--')
    axs[0].axvline(x=sbp_mean + sbp_std, ymin=0, ymax=1, color = 'red', linestyle = '--')
    axs[0].set_xlabel('SBP value [mmHg]')
    axs[0].set_ylabel('samples')
    axs[0].text(0.95, 0.95, 'Mean: %0.fmmHg\nStd: %0.fmmHg'%(sbp_mean, sbp_std), transform=axs[0].transAxes,
        fontsize=12, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs[1].hist(dist_data_dbp, bins=range(min(dist_data_dbp), max(dist_data_dbp)+2))
    axs[1].set_title(title + ' DBP')
    axs[1].axvline(x=dbp_mean, ymin=0, ymax=1, color = 'blue', linestyle = '--')
    axs[1].axvline(x=dbp_mean - sbp_std, ymin=0, ymax=1, color = 'red', linestyle = '--')
    axs[1].axvline(x=dbp_mean + sbp_std, ymin=0, ymax=1, color = 'red', linestyle = '--')
    axs[1].set_xlabel('DBP value [mmHg]')
    axs[1].set_ylabel('samples')
    axs[1].text(2.17, 0.95, 'Mean: %0.fmmHg\nStd: %0.fmmHg'%(dbp_mean, dbp_std), transform=axs[0].transAxes,
        fontsize=12, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    return fig_bp_dist, axs
  
  # absolute error
def error_plot(y_test, y_pred):

    y_test_2d = y_test[:,:,0]
    abs_error_sbp = y_test_2d.T[:,0] - y_pred[:,0]
    abs_error_dbp = y_test_2d.T[:,1] - y_pred[:,1]

    aberr_mean_sbp = np.mean(abs_error_sbp)
    aberr_mean_dbp = np.mean(abs_error_dbp)

    aberr_std_sbp = np.std(abs_error_sbp)
    aberr_std_dbp = np.std(abs_error_dbp)

    abs_error_sbp_int = abs_error_sbp.astype(int)
    abs_error_dbp_int = abs_error_dbp.astype(int)

    fig_abs_error, axs = plt.subplots(1, 2, figsize=(10, 5), tight_layout=False)

    axs[0].hist(abs_error_sbp_int, bins=range(min(abs_error_sbp_int), max(abs_error_sbp_int)+2))
    axs[0].set_title('BP-CRNN-50 SBP Error')
    axs[0].axvline(x=aberr_mean_sbp, ymin=0, ymax=1, color = 'blue', linestyle = '--')
    axs[0].axvline(x=aberr_mean_sbp - aberr_std_sbp, ymin=0, ymax=1, color = 'red', linestyle = '--')
    axs[0].axvline(x=aberr_mean_sbp + aberr_std_sbp, ymin=0, ymax=1, color = 'red', linestyle = '--')
    axs[0].set_xlim([-10, 60])
    axs[0].set_xlabel('error [mmHg]')
    axs[0].set_ylabel('samples')
    axs[0].text(0.95, 0.95, 'Mean: %0.2fmmHg\nStd: %0.2fmmHg'%(aberr_mean_sbp, aberr_std_sbp), transform=axs[0].transAxes,
        fontsize=12, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs[1].hist(abs_error_dbp_int, bins=range(min(abs_error_dbp_int), max(abs_error_dbp_int)+2))
    axs[1].set_title('BP-CRNN-50 DBP Error')
    axs[1].axvline(x=aberr_mean_dbp, ymin=0, ymax=1, color = 'blue', linestyle = '--')
    axs[1].axvline(x=aberr_mean_dbp - aberr_std_dbp, ymin=0, ymax=1, color = 'red', linestyle = '--')
    axs[1].axvline(x=aberr_mean_dbp + aberr_std_dbp, ymin=0, ymax=1, color = 'red', linestyle = '--')
    axs[1].set_xlim([-20, 20])
    axs[1].set_xlabel('error [mmHg]')
    axs[1].set_ylabel('samples')
    axs[1].text(2.17, 0.95, 'Mean: %0.2fmmHg\nStd: %0.2fmmHg'%(aberr_mean_dbp, aberr_std_dbp), transform=axs[0].transAxes,
        fontsize=12, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    return fig_abs_error, axs

def history_plot(hist, filename):

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), tight_layout=False)
    ax[0].plot(hist.history["mae"])
    ax[0].plot(hist.history["val_mae"])
    ax[0].set_title(filename + ' Model Mae')
    ax[0].set_ylabel("MAE")
    ax[0].set_xlabel("Epoch")
    ax[0].legend(['Train', 'Validation'], loc='upper left')
    ax[0].grid()
    
    ax[1].plot(hist.history['loss'])
    ax[1].plot(hist.history['val_loss'])
    ax[1].set_title(filename + ' Model loss')
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('epoch')
    ax[1].legend(['Train','Validation'], loc='upper right')
    ax[1].grid()

    return fig, ax

def bland_altman_plot(data1, data2, filename, *args, **kwargs):
    
    data1 = data1.T
    data2 = data2.T

    mean = np.mean([data1, data2], axis=0)
    diff = (data1 - data2)
    md = np.mean(diff, axis=1)
    sd = np.std(diff, axis=1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), tight_layout=False)

    ax[0].scatter(mean[0], diff[0], *args, **kwargs, alpha = 0.5)
    ax[0].axhline(md[0], color='gray', linestyle='--')
    ax[0].axhline(md[0] + 1.96*sd[0], color='red', linestyle='--')
    ax[0].axhline(md[0] - 1.96*sd[0], color='red', linestyle='--')
    ax[0].set_xlim([data1[0].min(), data1[0].max()])
    ax[0].set_ylim([-50, 50])
    ax[0].set_xlabel('Mean')
    ax[0].set_ylabel('Difference')
    ax[0].set_title(filename+' SBP Bland-Altman Plot')

    ax[1].scatter(mean[1], diff[1], *args, **kwargs, alpha = 0.5)
    ax[1].axhline(md[1], color='gray', linestyle='--')
    ax[1].axhline(md[1] + 1.96*sd[1], color='red', linestyle='--')
    ax[1].axhline(md[1] - 1.96*sd[1], color='red', linestyle='--')
    ax[1].set_xlim([data1[1].min(), data1[1].max()])
    ax[1].set_ylim([-50,50])
    ax[1].set_xlabel('Mean')
    ax[1].set_ylabel('Difference')
    ax[1].set_title(filename+' DBP Bland-Altman Plot')

    return fig, ax

def pred_plot(test_dat, pred_dat):
    
    sbp_min = test_dat[:, 0].min()
    sbp_max = test_dat[:, 0].max()
    dbp_min = test_dat[:, 1].min()
    dbp_max = test_dat[:, 1].max()
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), tight_layout=False)
    
    ax[0].scatter(test_dat[:,0], pred_dat[:,0], c='red', alpha = 0.5)
    ax[0].set_xlim([sbp_min-10, sbp_max+10])
    ax[0].set_ylim([sbp_min-10, sbp_max+10])
    ax[0].set_xlabel('True SBP')
    ax[0].set_ylabel('Estimated SBP')
    ax[0].set_title('SBP')
    
    ax[1].scatter(test_dat[:,1], pred_dat[:,1], c='red', alpha = 0.5)
    ax[1].set_xlim([dbp_min-10, dbp_max+10])
    ax[1].set_ylim([dbp_min-10, dbp_max+10])
    ax[1].set_xlabel('True DBP')
    ax[1].set_ylabel('Estimated DBP')
    ax[1].set_title('DBP')
    
    return fig, ax