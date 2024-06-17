import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model

def build_generator(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)

    encoder_layers = [
        (64, False),    
        (128, True),    
        (256, True),    
        (512, True),    
        (512, True),    
        (512, True),    
        (512, True),    
    ]

    skip_connections = []
    x = inputs

    for filters, apply_batchnorm in encoder_layers:
        x = Conv2D(filters, kernel_size=4, strides=2, padding='same')(x)
        if apply_batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        skip_connections.append(x)

    decoder_layers = [
        (512, True, 0.5),  
        (512, True, 0.5),
        (512, True, 0.5),
        (512, False, 0.0),
        (256, True, 0.5),
        (128, True, 0.0),
        (64, True, 0.0),
    ]

    decoder_layers.reverse()

    for idx, (filters, apply_dropout, dropout_rate) in enumerate(decoder_layers):
        x = Conv2DTranspose(filters, kernel_size=4, strides=2, padding='same')(x)
        if apply_dropout:
            x = Dropout(dropout_rate)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Concatenate()([x, skip_connections[-(idx + 1)]])  
    x = Conv2DTranspose(3, kernel_size=4, strides=2, padding='same')(x)
    outputs = Activation('tanh')(x)

    model = Model(inputs=inputs, outputs=outputs, name='generator')
    return model

def build_pix2pix():
    generator = build_generator()
    generator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5), loss='mae')
    return generator
