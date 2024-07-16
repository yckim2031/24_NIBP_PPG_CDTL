import os
import util
import argparse
import numpy as np
import train
import result
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model

parser = argparse.ArgumentParser()
# GPU config
parser.add_argument("-g_s", "--gpu_selection", type=str, 
                    default="-1", help="Type GPU number to be used")
parser.add_argument("-g_m", "--gpu_memory", type=int, default=0,
                    help="Type GPU occupancy [GB]")

# Dataset
parser.add_argument("--pretrained_model_dir",
                    default="/home/yckim/research/nibp_ppg/ex_3/model/collective/",
                    help="pretrained_model_dir")
parser.add_argument("--save_dir", 
                    default="/home/yckim/research/nibp_ppg/ex_3/model/transfer_test/",
                    help="save_folder_dir")
parser.add_argument("--dataset_dir", 
                    default='/home/yckim/research/nibp_ppg/ex_3/data/',
                    help="dataset_dir")
parser.add_argument("-s", "--source", type=str, default='', choices=['all','m1','m2','v1','v2', 'ra'], 
                    help="Select source group")
parser.add_argument("-t", "--target", type=str, default='', choices=['all','m1','m2','v1','v2', 'ra'], 
                    help="Select target group")

# Training config
parser.add_argument("--train_size", type=int, choices=[50, 100, 360, 720, 1800], 
                   default=50, help="Type data number to fine-tune")
parser.add_argument("-v", "--validation_size", type=int, default=720, 
                    help="Type valdiation data count")
parser.add_argument("--test_size", type=int, default=720, help="Type test data size")
parser.add_argument("-p", "--patience", type=int, default=2000, 
                    help="Set patience when training")
parser.add_argument("-e", "--epochs", type=int, default=20000, 
                    help="Set max epoch size")
parser.add_argument("-b", "--batch_size", type=int, default=32, 
                    help="Set batch size")

args = parser.parse_args()

util.gpu_config(args.gpu_selection, args.gpu_memory)

if __name__ == "__main__":
    
    #Directory path containing the .npz files
    pretrained_model_path = args.pretrained_model_dir+args.source
    save_folder_path = args.save_dir
    data_folder_path = args.dataset_dir+args.target
    
    #List all files in the folder
    pre_trained_model_list = util.find_files(pretrained_model_path, 'h5')
    data_list = util.find_files(data_folder_path, 'npz')
    data_cnt = args.train_size
    
    for pretrained_model in pre_trained_model_list:
        model_name = os.path.splitext(os.path.basename(pretrained_model))[0]
        target_list = [path for path in data_list if model_name not in path.split(os.path.sep)]
        for file_path in target_list:
            data = np.load(file_path)
            filename = os.path.splitext(os.path.basename(file_path))[0]
            x_data = data['x'][np.newaxis,:,:].transpose((1,2,0))
            y_data = data['y']
            if util.dataset_threshold(y_data, args) == False:
                continue
            
            attempt_cnt = 1
            
            for k in range(3): # 3 fold CV
            
                if attempt_cnt > 3:
                    continue
                      
                # define save folder path
                data_group_folder_name = model_name + "_to_" + file_path.split(os.path.sep)[-2]
                save_path = os.path.join(save_folder_path, data_group_folder_name, filename,
                                         str(data_cnt), str(attempt_cnt))+"/"
                util.make_save_dir(save_path)
                
                # Find any npz file exist in the target save folder path
                used_data_list = util.find_files(save_path, 'npz')
                if len(used_data_list)>0:
                    attempt_cnt += 1
                    print(f"The experiment on the subject {filename} is already done")
                    continue
    
                # Data selection
                train_indx, val_indx, test_indx = util.dataset_index(y_data.shape[0], attempt_cnt, args)
                x_train, x_val, x_test, y_train, y_val, y_test = x_data[train_indx], x_data[val_indx], x_data[test_indx], y_data[train_indx], y_data[val_indx], y_data[test_indx]
                
                # Model load & Fit
                train_args = {"x_train" : x_train, "y_train" : y_train, 
                              "x_val" : x_val, "y_val" : y_val, "patience" : args.patience, 
                              "epochs" : args.epochs, "batch_size" : args.batch_size}
                model = load_model(pretrained_model, compile = False)
                model = util.add_compile(model, learning_rate = 0.001)
                model, hist = train.train(model, train_args, trainable = [4, 5, 12, 13])
    
                # Predict and save results
                y_pred = model.predict(x_test)
                nan_array = np.isnan(y_pred)
                nan_indx = np.argwhere(nan_array ==True)
                print(nan_indx)
                nan_array = np.isnan(y_test)
                nan_indx = np.argwhere(nan_array ==True)
                print(nan_indx)
                sbp_mae, sbp_coef, dbp_mae, dbp_coef = result.mae_r(y_test, y_pred)
                result.texts(save_path, attempt_cnt, x_train, x_val, sbp_mae, sbp_coef, dbp_mae, dbp_coef)

                # figures
                fig_ba = util.bland_altman_plot(y_test, y_pred, filename, color='blue', marker='o')[0]
                fig_ba.savefig(save_path+filename+'_Bland_Altman.png')
                plt.close(fig_ba)
                
                fig_ac = util.pred_plot(y_test, y_pred)[0]
                fig_ac.savefig(save_path+filename+'_Pearson_R.png')
                plt.close(fig_ac)
                
                fig_history = util.history_plot(hist, filename)[0]
                fig_history.savefig(save_path+filename+'_History.png')
                plt.close(fig_history)
                
                # save used data as npz file
                np.savez(save_path+'used_data_'+str(attempt_cnt)+".npz", x_train=x_train, y_train=y_train, 
                         x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)
    
                model.save(save_path+filename + "_" + str(data_cnt) + "_" + 'tl_model.h5')
                
                attempt_cnt += 1
                
                tf.keras.backend.clear_session()
            
            data.close()
    
    print("Training completed")
