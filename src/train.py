import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.callbacks import EarlyStopping

def train(model, args, **kwargs):
        
    if 'trainable' in kwargs.keys():
        for m in range(len(model.layers)):
            if m in kwargs['trainable']:
                model.layers[m].trainable = True
            else:
                model.layers[m].trainable = False
    
    callback = EarlyStopping(monitor='val_mae', patience=args['patience'], 
                             restore_best_weights=True)
    
    hist = model.fit(args['x_train'], args['y_train'], epochs = args['epochs'], 
                     batch_size=args['batch_size'], validation_data = (args['x_val'], args['y_val']), 
                     verbose = 0, callbacks=[callback])
    
    return model, hist