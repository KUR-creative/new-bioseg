import timeit
import tensorflow as tf
import cv2
import numpy as np
from utils import decategorize, bgr_float32, bgr_uint8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

origin_map = {
    (0., 1.) : [1.0, 1.0, 1.0],
    (1., 0.) : [0.0, 0.0, 0.0]
}

def map_max_row(img, val=1):
    assert len(img.shape) == 3
    img2d = img.reshape(-1,img.shape[2])
    ret = np.zeros_like(img2d)
    ret[np.arange(len(img2d)), img2d.argmax(1)] = val
    return ret.reshape(img.shape)

def modulo_padded(img, modulo=16):
    h,w = img.shape[:2]
    h_padding = (modulo - (h % modulo)) % modulo
    w_padding = (modulo - (w % modulo)) % modulo
    if len(img.shape) == 3:
        return np.pad(img, [(0,h_padding),(0,w_padding),(0,0)], mode='reflect')
    elif len(img.shape) == 2:
        return np.pad(img, [(0,h_padding),(0,w_padding)], mode='reflect')

model_path = './fixture/snet.pb'
image_path = './fixture/0.png'
img = bgr_float32(cv2.imread(image_path))
oh,ow,oc = img.shape
img = modulo_padded(img)
h,w,c = img.shape

graph_def = tf.GraphDef()
start = timeit.default_timer() #-------------------------------------------------------
with tf.gfile.GFile(model_path, 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='snet')
stop  = timeit.default_timer() 
print('pb load time: ', stop - start)  #-----------------------------------------------

# print all op names in graph
#print(*[tensor.name for tensor in tf.get_default_graph().as_graph_def().node], sep='\n')


with tf.Session() as sess:
    snet_in  = sess.graph.get_tensor_by_name('snet/input_1:0')
    snet_out = sess.graph.get_tensor_by_name('snet/conv2d_19/truediv:0')
    # look and feel ops in graph with tensorboard
    #writer = tf.summary.FileWriter('./tmplog')
    #writer.add_graph(sess.graph)
    #writer.flush()
    #writer.close()

    print(h,w,c)
    start = timeit.default_timer() #-------------------------------------------------------
    out = sess.run(snet_out, {snet_in:img.reshape([1,h,w,c])})
    stop  = timeit.default_timer()
    print('pb run time: ', stop - start)  #-----------------------------------------------
    print(out.shape)
    mask = out.reshape(out.shape[1:]) 
    mask = map_max_row(mask)
    mask = decategorize( mask[:oh,:ow,:], origin_map )

    cv2.imshow('i', img)
    cv2.imshow('m', mask)
    cv2.waitKey(0)

'''
import keras
start = timeit.default_timer() #-------------------------------------------------------
snet = keras.models.load_model('./fixture/snet.h5', compile=False)
stop  = timeit.default_timer()
print('K load time: ', stop - start)  #-----------------------------------------------

start = timeit.default_timer() #-------------------------------------------------------
out  = snet.predict(img.reshape([1,h,w,c]), 1)
stop  = timeit.default_timer()
print('K run time: ', stop - start)  #-----------------------------------------------

mask2= out.reshape(out.shape[1:]) 
mask2= map_max_row(mask2)
mask2= decategorize( mask2[:oh,:ow,:], origin_map )
cv2.imshow('mk', mask2)

cv2.waitKey(0)
'''
