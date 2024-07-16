from tensorflow.keras import layers, models, Input

def set_layers(input):
    
    x1 = layers.Conv1D(filters = 50, kernel_size=7, activation='relu', padding='same')(input)
    x1 = layers.BatchNormalization()(x1)
    
    x2 = layers.Conv1D(filters = 50, kernel_size=7, activation='relu', padding='same')(x1)
    x2 = layers.BatchNormalization()(x2)
    
    x3 = layers.Conv1D(filters = 50, kernel_size=7, activation='relu', padding='same')(x2)
    concat = layers.Concatenate(axis = 2)([x1, x3])
    x4 = layers.BatchNormalization()(concat)
    
    x4 = layers.GRU(15, return_sequences=True, activation='tanh')(x4)
    x4 = layers.Flatten()(x4)
    x4 = layers.BatchNormalization()(x4)
    
    x4 = layers.Dense(units=32, activation='relu')(x4)
    x4 = layers.BatchNormalization()(x4)
    
    output = layers.Dense(units=2, activation='relu')(x4)
    
    return output

def build_network(**kwargs):
    input = Input(shape=(125,1))
    output = set_layers(input)
    model = models.Model(inputs = input, outputs = output)
    return model