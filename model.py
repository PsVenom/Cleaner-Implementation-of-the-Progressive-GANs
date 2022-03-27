import tensorflow as tf
from helper_functions import log2, num_filters, resolution_of_stage

## TODO: Add a class to scale the outputs from the Convolution layer Here

## Convolution Layer
def standard_conv(filters, kernel_size, padding = "same", strides = 1, activation = tf.nn.leaky_relu):
    return tf.keras.layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  strides=strides,
                                  activation=activation)

## Returns a Convolutional layer with 3 Filters
def rgb_conv(kernel_size, padding = "same", strides = 1, filters = 3):
    return tf.keras.layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  strides=strides,)


## Pixel Normalization Layer
class PixelNorm(tf.keras.layers.Layer):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def call(self, x, **kwargs):
        return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + self.epsilon)


## TODO: Edit the ConvBlock Class to suit the discriminator architecture
## Convolution block
class ConvBlock(tf.keras.layers.Conv2D):
    def __init__(self, filters, use_pixelnorm = True):
        super(ConvBlock, self).__init__()
        self.use_pn = use_pixelnorm
        self.conv1 = standard_conv(filters, kernel_size=(3, 3))
        self.conv2 = standard_conv(filters, kernel_size=(1, 1))
        self.pn = PixelNorm()

    def call(self, x):
        x = self.conv1(x)
        x = self.pn(x) if self.use_pn else x
        x = self.conv2(x)
        x = self.pn(x) if self.use_pn else x
        return x


### TODO: Complete the generator block, design it such that it can adapt to the resolution.
## Generator Block
class GeneratorBlock:
    pass


### TODO:  Complete the Generator class such that:
### TODO: 1) It can send an output of the initial blocks before growing (COMPLETE)
### TODO: 2) It can grow progressively (Add GeneratorBlocks to it's layers outputting double the resolution of images)
## Building the generator
class Generator(tf.keras.models.Sequential):
    def __init__(self, latent_dim = 512, normalize_latents = True,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage = 0
        self.normalize_latents = normalize_latents
        ### Initial block taking 1x1 to 4x4
        self.initial_layers = tf.keras.Sequential(
            [
                PixelNorm(),
                tf.keras.layers.Conv2DTranspose(filters=num_filters(self.stage), kernel_size=4),
                standard_conv(filters=num_filters(self.stage),
                              kernel_size=3),
                PixelNorm()
            ]
        )
        ## Initial RGB layer
        self.initial_rgb = rgb_conv(kernel_size=1)
        self.prog_block = [tf.keras.layers.InputLayer([latent_dim]), self.initial_layers]
        self.resolution: int = resolution_of_stage(self.stage)

    def call(self, inputs, training=None, mask=None):
        print("def call was called inside the Generator Class")
        x = tf.nn.l2_normalize(inputs, axis = -1) if self.normalize_latents else inputs
        for layer in self.prog_block:
            x = layer(x)
        return x

# noise = tf.random.normal([1, 1, 1, 128])
# gen_1 = Generator()
# out = gen_1(noise)
# print(out.shape)
# gen_1.initial_layers.summary()








