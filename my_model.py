from keras.models import *
from keras.layers import *

def layer_BN_relu(input,layer_fn,*args,**kargs):
    x = layer_fn(*args,**kargs)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def layer_relu(input,layer_fn,*args,**kargs):
    x = layer_fn(*args,**kargs)(input)
    x = Activation('relu')(x)
    return x

def down_block(x, cnum, kernel_init, filter_vec=(3,3,1), maxpool2x=True, 
               kernel_regularizer=None, bias_regularizer=None,
               basic_layer=layer_BN_relu):
    if all(isinstance(x,int) for x in filter_vec):
        for n in filter_vec:
            x = basic_layer(
                    x, Conv2D, cnum, (n,n), 
                    padding='same', kernel_initializer=kernel_init,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
    else:
        for n,BN in filter_vec:
            layer = layer_BN_relu if BN else layer_relu
            x = layer(
                    x, Conv2D, cnum, (n,n), 
                    padding='same', kernel_initializer=kernel_init,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
    if maxpool2x:
        pool = MaxPooling2D(pool_size=(2,2))(x)
        return x, pool
    else:
        return x

def up_block(from_horizon, upward, cnum, kernel_init, filter_vec=(3,3,1), 
             kernel_regularizer=None, bias_regularizer=None,
             basic_layer=layer_BN_relu):
    upward = Conv2DTranspose(cnum, (2,2), padding='same', strides=(2,2), 
                 kernel_initializer=kernel_init,
                 kernel_regularizer=kernel_regularizer,
                 bias_regularizer=bias_regularizer)(upward)
    merged = concatenate([from_horizon,upward], axis=3)
    if all(isinstance(x,int) for x in filter_vec):
        for n in filter_vec:
            merged = basic_layer(
                merged, Conv2D, cnum, (n,n), padding='same', 
                kernel_initializer=kernel_init,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
    else:
        for n,BN in filter_vec:
            layer = layer_BN_relu if BN else layer_relu
            merged = layer(
                merged, Conv2D, cnum, (n,n), padding='same', 
                kernel_initializer=kernel_init,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
    return merged

def unet(input_size = (None,None,3), pretrained_weights = None,
         kernel_init='he_normal', 
         num_classes=3, last_activation='softmax',
         num_filters=64, num_maxpool = 4, filter_vec=(3,3,1),
         kernel_regularizer=None, bias_regularizer=None,
         dropout=None, basic_layer=layer_BN_relu):
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
                                      bias_regularizer=bias_regularizer,
                                      basic_layer=basic_layer)

    x = down_block(x, 2**depth * cnum, kernel_init, filter_vec=filter_vec, maxpool2x=False,
		   kernel_regularizer=kernel_regularizer,
		   bias_regularizer=bias_regularizer,
                   basic_layer=basic_layer)

    x = Dropout(dropout)(x) if dropout else x

    for i in reversed(range(depth)): 
        x = up_block(down_convs[i], x, 2**i * cnum, kernel_init, filter_vec=filter_vec,
		     kernel_regularizer=kernel_regularizer,
		     bias_regularizer=bias_regularizer,
                     basic_layer=basic_layer)

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
