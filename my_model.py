from keras.models import *
from keras.layers import *

def set_layer_BN_relu(input,layer_fn,*args,**kargs):
    x = layer_fn(*args,**kargs)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def down_block(x, cnum, kernel_init, filter_vec=(3,3,1), maxpool2x=True, 
               kernel_regularizer=None, bias_regularizer=None):
    for n in filter_vec:
        x = set_layer_BN_relu(
                x, Conv2D, cnum, (n,n), 
                padding='same', kernel_initializer=kernel_init,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
    if maxpool2x:
        pool = MaxPooling2D(pool_size=(2,2))(x)
        return x, pool
    else:
        return x

def up_block(from_horizon, upward, cnum, kernel_init, filter_vec=(3,3,1), 
             kernel_regularizer=None, bias_regularizer=None):
    upward = Conv2DTranspose(cnum, (2,2), padding='same', strides=(2,2), 
                 kernel_initializer=kernel_init,
                 kernel_regularizer=kernel_regularizer,
                 bias_regularizer=bias_regularizer)(upward)
    merged = concatenate([from_horizon,upward], axis=3)
    for n in filter_vec:
        merged = set_layer_BN_relu(
            merged, Conv2D, cnum, (n,n), padding='same', 
            kernel_initializer=kernel_init,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
    return merged

def unet(input_size = (None,None,3), pretrained_weights = None,
         kernel_init='he_normal', 
         num_classes=3, last_activation='softmax',
         num_filters=64, num_maxpool = 4, filter_vec=(3,3,1),
         kernel_regularizer=None, bias_regularizer=None,
         dropout=None):
    '''
    depth = 4
    inp -> 0-------8 -> out
            1-----7
             2---6
              3-5
               4     <--- dropout 
    '''
    cnum = num_filters
    depth = num_maxpool

    x = inp = Input(input_size)

    down_convs = [None] * depth
    for i in range(depth): 
        down_convs[i], x = down_block(x, 2**i * cnum, kernel_init, filter_vec=filter_vec, 
                                      kernel_regularizer=kernel_regularizer,
                                      bias_regularizer=bias_regularizer)

    x = down_block(x, 2**depth * cnum, kernel_init, filter_vec=filter_vec, maxpool2x=False,
		   kernel_regularizer=kernel_regularizer,
		   bias_regularizer=bias_regularizer)

    x = Dropout(dropout)(x) if dropout else x

    for i in reversed(range(depth)): 
        x = up_block(down_convs[i], x, 2**i * cnum, kernel_init, filter_vec=filter_vec,
		     kernel_regularizer=kernel_regularizer,
		     bias_regularizer=bias_regularizer)

    #print('nc:',num_classes, 'la:',last_activation)
    if last_activation == 'sigmoid':
        out_channels = 1
    else:
        out_channels = num_classes
    out = Conv2D(out_channels, (1,1), padding='same',
                 kernel_initializer=kernel_init, activation = last_activation)(x)

    model = Model(input=inp, output=out)

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
