import tensorflow as tf

NUM_CLASSES = 1000

def fire_module(x,inp,sp,e11p,e33p):
    with tf.compat.v1.variable_scope("fire"):
        with tf.compat.v1.variable_scope("squeeze"):
            W = tf.compat.v1.get_variable("weights",shape=[1,1,inp,sp])
            b = tf.compat.v1.get_variable("bias",shape=[sp])
            s = tf.compat.v1.nn.conv2d(x,W,[1,1,1,1],"VALID")+b
            s = tf.compat.v1.nn.relu(s)
        with tf.compat.v1.variable_scope("e11"):
            W = tf.compat.v1.get_variable("weights",shape=[1,1,sp,e11p])
            b = tf.compat.v1.get_variable("bias",shape=[e11p])
            e11 = tf.compat.v1.nn.conv2d(s,W,[1,1,1,1],"VALID")+b
            e11 = tf.compat.v1.nn.relu(e11)
        with tf.compat.v1.variable_scope("e33"):
            W = tf.compat.v1.get_variable("weights",shape=[3,3,sp,e33p])
            b = tf.compat.v1.get_variable("bias",shape=[e33p])
            e33 = tf.compat.v1.nn.conv2d(s,W,[1,1,1,1],"SAME")+b
            e33 = tf.compat.v1.nn.relu(e33)
        return tf.compat.v1.concat([e11,e33],3)


class SqueezeNet(object):
    def extract_features(self, input=None, reuse=True):
        if input is None:
            input = self.image
        x = input
        layers = []
        with tf.compat.v1.variable_scope('features', reuse=reuse):
            with tf.compat.v1.variable_scope('layer0'):
                W = tf.compat.v1.get_variable("weights",shape=[3,3,3,64])
                b = tf.compat.v1.get_variable("bias",shape=[64])
                x = tf.compat.v1.nn.conv2d(x,W,[1,2,2,1],"VALID")
                x = tf.compat.v1.nn.bias_add(x,b)
                layers.append(x)
            with tf.compat.v1.variable_scope('layer1'):
                x = tf.compat.v1.nn.relu(x)
                layers.append(x)
            with tf.compat.v1.variable_scope('layer2'):
                x = tf.compat.v1.nn.max_pool(x,[1,3,3,1],strides=[1,2,2,1],padding='VALID')
                layers.append(x)
            with tf.compat.v1.variable_scope('layer3'):
                x = fire_module(x,64,16,64,64)
                layers.append(x)
            with tf.compat.v1.variable_scope('layer4'):
                x = fire_module(x,128,16,64,64)
                layers.append(x)
            with tf.compat.v1.variable_scope('layer5'):
                x = tf.compat.v1.nn.max_pool(x,[1,3,3,1],strides=[1,2,2,1],padding='VALID')
                layers.append(x)
            with tf.compat.v1.variable_scope('layer6'):
                x = fire_module(x,128,32,128,128)
                layers.append(x)
            with tf.compat.v1.variable_scope('layer7'):
                x = fire_module(x,256,32,128,128)
                layers.append(x)
            with tf.compat.v1.variable_scope('layer8'):
                x = tf.compat.v1.nn.max_pool(x,[1,3,3,1],strides=[1,2,2,1],padding='VALID')
                layers.append(x)
            with tf.compat.v1.variable_scope('layer9'):
                x = fire_module(x,256,48,192,192)
                layers.append(x)
            with tf.compat.v1.variable_scope('layer10'):
                x = fire_module(x,384,48,192,192)
                layers.append(x)
            with tf.compat.v1.variable_scope('layer11'):
                x = fire_module(x,384,64,256,256)
                layers.append(x)
            with tf.compat.v1.variable_scope('layer12'):
                x = fire_module(x,512,64,256,256)
                layers.append(x)
        return layers

    def __init__(self, save_path=None, sess=None):
        """Create a SqueezeNet model.
        Inputs:
        - save_path: path to TensorFlow checkpoint
        - sess: TensorFlow session
        - input: optional input to the model. If None, will use placeholder for input.
        """
        self.image = tf.compat.v1.placeholder('float',shape=[None,None,None,3],name='input_image')
        self.labels = tf.compat.v1.placeholder('int32', shape=[None], name='labels')
        self.layers = []
        x = self.image
        self.layers = self.extract_features(x, reuse=False)
        self.features = self.layers[-1]
        with tf.compat.v1.variable_scope('classifier'):
            with tf.compat.v1.variable_scope('layer0'):
                x = self.features
                self.layers.append(x)
            with tf.compat.v1.variable_scope('layer1'):
                W = tf.compat.v1.get_variable("weights",shape=[1,1,512,1000])
                b = tf.compat.v1.get_variable("bias",shape=[1000])
                x = tf.compat.v1.nn.conv2d(x,W,[1,1,1,1],"VALID")
                x = tf.compat.v1.nn.bias_add(x,b)
                self.layers.append(x)
            with tf.compat.v1.variable_scope('layer2'):
                x = tf.compat.v1.nn.relu(x)
                self.layers.append(x)
            with tf.compat.v1.variable_scope('layer3'):
                x = tf.compat.v1.nn.avg_pool(x,[1,13,13,1],strides=[1,13,13,1],padding='VALID')
                self.layers.append(x)
        self.classifier = tf.compat.v1.reshape(x,[-1, NUM_CLASSES])

        if save_path is not None:
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, save_path)
        self.loss = tf.compat.v1.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits(labels=tf.compat.v1.one_hot(self.labels, NUM_CLASSES), logits=self.classifier))
