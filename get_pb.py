from keras.models import load_model

import tensorflow as tf
import keras.backend as K
from keras.models import Model
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.python.tools.freeze_graph import freeze_graph_with_def_protos

def serialize(model, output_path):
    """
    https://gist.github.com/bobpoekert/55136024048075989d283192badac0a0
    model: a keras Model
    output_path: filepath to save the tensorflow pb model to
    """
    session = tf.Session()
    K.set_session(session)
    K.set_learning_phase(0)

    # prevent error: Attempting to use uninitialized value batch_normalization_3/gamma
    # https://stackoverflow.com/questions/41818654/keras-batchnormalization-uninitialized-value
    session.run(tf.global_variables_initializer())

    config = model.get_config()
    weights = model.get_weights()

    input_graph = session.graph_def
    output_graph = graph_util.convert_variables_to_constants(
            session, input_graph, [v.name.split(':')[0] for v in model.outputs])

    with open(output_path, 'wb') as outf:
        outf.write(output_graph.SerializeToString())

snet = load_model('./fixture/snet.h5', compile=False)
serialize(snet, './fixture/test.pb')
