import tensorflow as tf
import numpy as np


def relu(x): return tf.keras.layers.Activation('relu')(x)

def conv_costum_init_layer(nf, ks, name, weight_decay, stddev_init):
    kernel_reg = tf.keras.regularizers.l2(weight_decay[0]) if weight_decay else None
    bias_reg = tf.keras.regularizers.l2(weight_decay[1]) if weight_decay else None

    x = tf.keras.layers.Conv2D(nf, (ks, ks), padding='same', name=name,
                               kernel_regularizer=kernel_reg,
                               bias_regularizer=bias_reg,
                               kernel_initializer=tf.keras.initializers.random_normal(stddev=stddev_init),
                               bias_initializer=tf.keras.initializers.constant(0.0))
    return x

class DenseLayer(tf.keras.layers.Layer):

    def __init__(self, grow_rate, features_count_in, name, weight_decay, training):
        super(DenseLayer, self).__init__()

        self.features_count = features_count_in + grow_rate
        self.training = training
        kernel_reg = tf.keras.regularizers.l2(weight_decay[0]) if weight_decay else None
        bias_reg = tf.keras.regularizers.l2(weight_decay[1]) if weight_decay else None

        self.batch_norm_layer = tf.keras.layers.BatchNormalization(name=name+'_bn')
        self.conv_layer = tf.keras.layers.Conv2D(grow_rate, (3, 3), padding='same', name=name+'_conv',
                                                 kernel_regularizer=kernel_reg,
                                                 bias_regularizer=bias_reg,
                                                 kernel_initializer=tf.keras.initializers.random_normal(stddev=np.sqrt(1.0 / features_count_in)),
                                                 bias_initializer=tf.keras.initializers.constant(0.0))
        self.concat_layer = tf.keras.layers.Concatenate(axis=-1)

    def call(self, input):

        x = self.batch_norm_layer(input, training=self.training)
        x = relu(x)
        x = self.conv_layer(x)
        x = self.concat_layer([input, x])

        return x



class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, layer_num, features_count_in, name, params, training):
        super(DenseBlock, self).__init__()

        self.layer_num = layer_num
        self.features_count = features_count_in
        self.layers = []

        grow_rate = params['grow_rate']
        weight_decay = params['weight_decay']

        for it in range(self.layer_num):
            layer_name = name + '_' + str(it)
            self.layers.append(DenseLayer(grow_rate, self.features_count, layer_name, weight_decay, training))
            self.features_count = self.layers[it].features_count

    def call(self, input):

        x = input

        for it in range(self.layer_num):
            layer_it = self.layers[it]
            x = layer_it(x)

        return x


class TransitionLayerUp(tf.keras.layers.Layer):
    def __init__(self, features_count_in, name, weight_decay, padding, training):
        super(TransitionLayerUp, self).__init__()

        self.features_count = features_count_in
        self.training = training

        kernel_reg = tf.keras.regularizers.l2(weight_decay[0]) if weight_decay else None
        bias_reg = tf.keras.regularizers.l2(weight_decay[1]) if weight_decay else None

        self.conv_transpose =  tf.keras.layers.Conv2DTranspose(features_count_in, (3, 3), name=name + '_conv_transpose',
                                                               padding=padding,
                                                               strides=(2,2),
                                                               kernel_regularizer=kernel_reg,
                                                               bias_regularizer=bias_reg,
                                                               kernel_initializer=tf.keras.initializers.he_normal(),
                                                               bias_initializer=tf.keras.initializers.constant(0.0))

    def call(self, input):
        x = self.conv_transpose(input)

        return x


class DenseLayerFCNet(tf.keras.layers.Layer):

    def __init__(self, grow_rate, features_count_in, name, weight_decay, training):
        super(DenseLayerFCNet, self).__init__()

        self.features_count = grow_rate
        self.training = training
        kernel_reg = tf.keras.regularizers.l2(weight_decay[0]) if weight_decay else None
        bias_reg = tf.keras.regularizers.l2(weight_decay[1]) if weight_decay else None

        self.batch_norm_layer = tf.keras.layers.BatchNormalization(name=name+'_bn')
        self.conv_layer = tf.keras.layers.Conv2D(grow_rate, (3, 3), padding='same', name=name+'_conv',
                                                 kernel_regularizer=kernel_reg,
                                                 bias_regularizer=bias_reg,
                                                 kernel_initializer=tf.keras.initializers.he_normal(),
                                                 bias_initializer=tf.keras.initializers.constant(0.0))

    def call(self, input):

        x = self.batch_norm_layer(input, training=self.training)
        x = relu(x)
        x = self.conv_layer(x)

        return x


class DenseBlockFCNet(tf.keras.layers.Layer):
    def __init__(self, layer_num, features_count_in, name, params, training):
        super(DenseBlockFCNet, self).__init__()

        grow_rate = params['grow_rate']
        weight_decay = params['weight_decay']

        self.layer_num = layer_num
        self.features_count = layer_num * grow_rate
        self.layers = []
        self.concat_layers = []

        for it in range(self.layer_num):
            layer_name = name + '_' + str(it)
            self.layers.append(DenseLayerFCNet(grow_rate, features_count_in, layer_name, weight_decay, training))
            features_count_in = features_count_in + grow_rate
            self.concat_layers.append(tf.keras.layers.Concatenate(axis=-1))

        self.out_concat_layer = tf.keras.layers.Concatenate(axis=-1)

    def call(self, input):

        layer_out_list = []

        dense_layer_in = input

        for it in range(self.layer_num):
            layer_it = self.layers[it]
            layer_out = layer_it(dense_layer_in)
            concat_it = self.concat_layers[it]
            dense_layer_in = concat_it([dense_layer_in, layer_out])
            layer_out_list.append(layer_out)

        x = self.out_concat_layer(layer_out_list)

        return x



class TransitionLayerMaxPoolingNoConvLayers(tf.keras.layers.Layer):
    def __init__(self, features_count_in, name, weight_decay, training):
        super(TransitionLayerMaxPoolingNoConvLayers, self).__init__()

        self.features_count = features_count_in
        self.training = training

        self.max_pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")

    def call(self, input):
        x = self.max_pooling(input)

        return x



class UDenseNet(tf.keras.layers.Layer):
    def __init__(self, name, params, training):
        super(UDenseNet, self).__init__()

        self.features_count = 48

        weight_decay = params['weight_decay']
        kernel_reg = tf.keras.regularizers.l2(weight_decay[0]) if weight_decay else None
        bias_reg = tf.keras.regularizers.l2(weight_decay[1]) if weight_decay else None

        self.conv_layer = tf.keras.layers.Conv2D(self.features_count, (3, 3), padding='same', name=name+'_conv',
                                                 kernel_regularizer=kernel_reg,
                                                 bias_regularizer=bias_reg,
                                                 kernel_initializer=tf.keras.initializers.random_normal(
                                                     stddev=np.sqrt(1.0 / 6)),
                                                 bias_initializer=tf.keras.initializers.constant(0.0))


        # block 1
        self.dense_block_1 = DenseBlock(layer_num=4, features_count_in=self.features_count, name=name+'dense_block_1',
                                        params=params, training=training)
        self.features_count = self.dense_block_1.features_count
        self.transition_layer_1 = TransitionLayerMaxPoolingNoConvLayers(features_count_in=self.features_count,
                                                                        name=name+'transition_layer_1',
                                                                        weight_decay=weight_decay,
                                                                        training=training)
        self.features_count = self.transition_layer_1.features_count

        # block 2
        self.dense_block_2 = DenseBlock(layer_num=5, features_count_in=self.features_count, name=name+'dense_block_2',
                                        params=params, training=training)
        self.features_count = self.dense_block_2.features_count
        self.transition_layer_2 = TransitionLayerMaxPoolingNoConvLayers(features_count_in=self.features_count,
                                                                        name=name+'transition_layer_2',
                                                                        weight_decay=weight_decay,
                                                                        training=training)
        self.features_count = self.transition_layer_2.features_count

        # block 3
        self.dense_block_3 = DenseBlock(layer_num=7, features_count_in=self.features_count, name=name+'dense_block_3',
                                        params=params, training=training)
        self.features_count = self.dense_block_3.features_count
        features_count_encoder_block_in = self.features_count

        # encoder block 1
        self.transition_layer_endcoder_1 = TransitionLayerMaxPoolingNoConvLayers(features_count_in=self.features_count,
                                                                                 name='transition_layer_endcoder_1',
                                                                                 weight_decay=weight_decay,
                                                                                 training=training)
        self.features_count = self.transition_layer_endcoder_1.features_count
        self.dense_block_endcoder_1 = DenseBlock(layer_num=10, features_count_in=self.features_count, name="dense_block_endcoder_1",
                                                 params=params, training=training)
        self.features_count = self.dense_block_endcoder_1.features_count
        features_count_encoder_block_1 = self.features_count

        # encoder block 2
        self.transition_layer_endcoder_2 = TransitionLayerMaxPoolingNoConvLayers(features_count_in=self.features_count,
                                                                                 name='transition_layer_endcoder_2',
                                                                                 weight_decay=weight_decay,
                                                                                 training=training)
        self.features_count = self.transition_layer_endcoder_2.features_count

        self.dense_block_endcoder_2 = DenseBlock(layer_num=12, features_count_in=self.features_count, name="dense_block_endcoder_2",
                                                 params=params, training=training)
        self.features_count = self.dense_block_endcoder_2.features_count
        features_count_encoder_block_2 = self.features_count

        # bottleneck
        self.transition_layer_bottleneck = TransitionLayerMaxPoolingNoConvLayers(features_count_in=self.features_count,
                                                                                 name='transition_layer_bottleneck',
                                                                                 weight_decay=weight_decay,
                                                                                 training=training)
        self.features_count = self.transition_layer_bottleneck.features_count
        self.dense_block_bottleneck = DenseBlockFCNet(layer_num=15, features_count_in=self.features_count, name="dense_block_bottleneck",
                                                      params=params, training=training)
        self.features_count = self.dense_block_bottleneck.features_count

        # decoder block 2
        self.transition_layer_decoder_2 = TransitionLayerUp(features_count_in=self.features_count,
                                                            name="transition_layer_decoder_2",
                                                            weight_decay=weight_decay,
                                                            padding='valid',
                                                            training=training)
        self.features_count = self.transition_layer_decoder_2.features_count
        self.concat_decoder_block_2 = tf.keras.layers.Concatenate(axis=-1)
        self.features_count = self.features_count + features_count_encoder_block_2
        self.dense_block_decoder_2 = DenseBlockFCNet(layer_num=12, features_count_in=self.features_count, name="dense_block_decoder_2",
                                                     params=params, training=training)
        self.features_count = self.dense_block_decoder_2.features_count

        # decoder block 1
        self.transition_layer_decoder_1 = TransitionLayerUp(features_count_in=self.features_count,
                                                            name="transition_layer_decoder_1",
                                                            weight_decay=weight_decay,
                                                            padding='valid',
                                                            training=training)
        self.features_count = self.transition_layer_decoder_1.features_count
        self.concat_decoder_block_1 = tf.keras.layers.Concatenate(axis=-1)
        self.features_count = self.features_count + features_count_encoder_block_1
        self.dense_block_decoder_1 = DenseBlockFCNet(layer_num=10, features_count_in=self.features_count, name="dense_block_decoder_1",
                                                     params=params, training=training)
        self.features_count = self.dense_block_decoder_1.features_count

        # decoder block out
        self.transition_layer_decoder_out = TransitionLayerUp(features_count_in=self.features_count,
                                                              name="transition_layer_decoder_out",
                                                              weight_decay=weight_decay,
                                                              padding='same',
                                                              training=training)
        self.features_count = self.transition_layer_decoder_out.features_count
        self.concat_decoder_block_with_in = tf.keras.layers.Concatenate(axis=-1)
        self.features_count = self.features_count + features_count_encoder_block_in
        self.dense_block_decoder_out = DenseBlockFCNet(layer_num=7, features_count_in=self.features_count, name="dense_block_decoder_out",
                                                       params=params, training=training)
        self.concat_decoder_block_out = tf.keras.layers.Concatenate(axis=-1)
        self.features_count = self.features_count + self.dense_block_decoder_out.features_count


    def call(self, input):

        x = self.conv_layer(input)

        x = self.dense_block_1(x)
        x = self.transition_layer_1(x)

        x = self.dense_block_2(x)
        x = self.transition_layer_2(x)

        x = self.dense_block_3(x)

        encoder_input = x
        x = self.transition_layer_endcoder_1(encoder_input)
        x = self.dense_block_endcoder_1(x)
        encoder_block_1_out = x

        x = self.transition_layer_endcoder_2(x)
        x = self.dense_block_endcoder_2(x)
        encoder_block_2_out = x

        x = self.transition_layer_bottleneck(x)
        x = self.dense_block_bottleneck(x)

        bottleneck_out = x

        x = self.transition_layer_decoder_2(x)
        x = self.concat_decoder_block_2([encoder_block_2_out, x])
        x = self.dense_block_decoder_2(x)

        decoder_2_out = x

        x = self.transition_layer_decoder_1(x)
        x = self.concat_decoder_block_1([encoder_block_1_out, x])
        x = self.dense_block_decoder_1(x)

        decoder_1_out = x

        x = self.transition_layer_decoder_out(x)
        x_dense_block_out_in = self.concat_decoder_block_with_in([encoder_input, x])
        x = self.dense_block_decoder_out(x_dense_block_out_in)
        x = self.concat_decoder_block_out([x_dense_block_out_in, x])

        return x, bottleneck_out, decoder_2_out, decoder_1_out