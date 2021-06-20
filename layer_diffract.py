import numpy as np
import tensorflow as tf
from keras.layers import Lambda

# 再生計算
def diff_layer(x, wl=633e-9, p=10e-6, z=-0.2):
        shape=x.get_shape().as_list()
        batch=shape[0]
        nx=shape[2]
        ny=shape[1]
        ch=shape[3]
        nx2=nx*2
        ny2=ny*2
        px=1/(nx2*p)
        py=1/(ny2*p)

        print("diffract_layer shape : ", x)
        f=tf.squeeze(x, axis=-1)
        print("diffract_layer shape : ", f)

        ### zero padding
        f=Lambda(lambda v: tf.pad(v,[[0,0],[0,ny],[0,nx]]))(f)

        # print("xxx before:",f.shape)
        f = Lambda(lambda v: tf.signal.fft2d(tf.cast(v, tf.complex64)))(f)

        print("diffract_layer shape : ", f)

        ### generate transfer function of ASM
        print("diffract_layer shape : ", f)
                
        x, y = tf.meshgrid(tf.linspace(-ny2/2+1, ny2/2, tf.cast(ny2, tf.int32)),
                           tf.linspace(-nx2/2+1, nx2/2, tf.cast(nx2, tf.int32)))
        fx = tf.cast(x,tf.float32)*px
        fy = tf.cast(y,tf.float32)*py
        ph=tf.exp(tf.dtypes.complex(tf.dtypes.cast(0.,tf.float32),+2*np.pi*z*tf.sqrt(1/(wl*wl)-(fx**2+fy**2))))
        ph=tf.signal.fftshift(ph)

        f = Lambda(lambda v: tf.math.multiply(v[0], v[1]))([f,ph]) 
        f = Lambda(lambda v: tf.signal.ifft2d(tf.cast(v, tf.complex64)))(f)

        f = Lambda(lambda v:  tf.slice(v,(0,0,0),(-1,ny,nx)))(f)

        f = Lambda(lambda v: tf.abs(v))(f)
        f = Lambda(lambda v: tf.pow(v, 2.0))(f)
        f = Lambda(lambda v: tf.cast(v,tf.float32))(f)

        f=tf.expand_dims(f,axis=-1)

        return f