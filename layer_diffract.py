import numpy as np
import tensorflow as tf
from keras.layers import Lambda
import numpy as np
import math

# 再生計算
def diff_layer(args, wl=633e-9, p=10e-6): #z=default:-0.2/best:-0.04
    x = args[0]
    z = args[1]
    shape = x.get_shape().as_list()
    batch = shape[0]
    nx = shape[2]
    ny = shape[1]
    ch = shape[3]
    nx2 = nx*2
    ny2 = ny*2
    px = 1/(nx2*p)
    py = 1/(ny2*p)

    # print("diffract_layer shape : ", x)
    f = tf.squeeze(x, axis=-1) # size が 1 の指定された次元を消す
    # print("diffract_layer shape : ", f)

    # zero padding
    f = Lambda(lambda v: tf.pad(v, [[0, 0], [0, ny], [0, nx]]))(f)

    # print("xxx before:",f.shape)
    f = Lambda(lambda v: tf.signal.fft2d(tf.cast(v, tf.complex64)))(f) # フーリエ変換

    # generate transfer function of ASM
    # print("diffract_layer shape : ", f)

    x, y = tf.meshgrid(tf.linspace(-ny2/2+1, ny2/2, tf.cast(ny2, tf.int32)),
                       tf.linspace(-nx2/2+1, nx2/2, tf.cast(nx2, tf.int32)))
    fx = tf.cast(x, tf.float32)*px # tf.castはxをtf.float32に型変換
    fy = tf.cast(y, tf.float32)*py
    ph = tf.exp(tf.dtypes.complex(tf.dtypes.cast(0., tf.float32), +
                                  2*np.pi*z*tf.sqrt(1/(wl*wl)-(fx**2+fy**2))))
    ph = tf.signal.fftshift(ph) # phをシフト演算

    f = Lambda(lambda v: tf.math.multiply(v[0], v[1]))([f, ph]) # G2=G1*H (f=f*ph)
    f = Lambda(lambda v: tf.signal.ifft2d(tf.cast(v, tf.complex64)))(f) # 逆フーリエ変換

    f = Lambda(lambda v:  tf.slice(v, (0, 0, 0), (-1, ny, nx)))(f)

    f = Lambda(lambda v: tf.abs(v))(f) # g2の振幅を求める？
    f = Lambda(lambda v: tf.pow(v, 2.0))(f) # 振幅を2乗
    f = Lambda(lambda v: tf.cast(v, tf.float32))(f)

    f = tf.expand_dims(f, axis=-1) # 1つ次元を追加

    return f

# 0～255に正規化
def normalize(x, nx, ny):
	x = x[0]
	x = x[:,:,0]
	print("normalize_x_shape : ", x.shape)
	x_min = x[0][0]
	x_max = x[0][0]
	# max, minを求める
	for ax in range(nx):
		for ay in range(ny):
			if x_min > x[ax][ay]:
				x_min = x[ax][ay]
			if x_max < x[ax][ay]:
				x_max = x[ax][ay]
	# 0～255に正規化
	for ax in range(nx):
		for ay in range(ny):
			x[ax][ay] = (255 * (x[ax][ay] - x_min) / (x_max - x_min))
	x = x[np.newaxis, ...]
	x = x.reshape(1,nx,ny,1)
	print("normalize_x_shape : ", x.shape)
	return x

# def diff_layer2(x, nx, ny, wl=633e-9, p=10e-6, z=-80): #z=-0.2
# 	x = np.fft.fft(x)
# 	for ax in range(nx):
# 		for ay in range(ny):
# 			abs_x = abs(x[ax][ay])
# 			x[ax][ay] = math.log10(2+1)
# 	return x