import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Add, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

from keras.layers import Lambda
from tensorflow.python.keras.backend import conv2d
import layer_diffract as ld

# U-Netのモデルを返す 再生画像で学習
def unet(input_shape=(1, 128, 1) ):
    filt_size=3
    rate=1
    resnet_flag=0
    n_filt=8
    img_input = Input(shape=input_shape)
    skip=img_input

    # ダウンサンプリング
    m1 = Conv2D(n_filt, filt_size, padding='same')(img_input)
    m1=LeakyReLU()(m1)
    p1 = MaxPooling2D(pool_size=(2, 2))(m1)
    
    m2 = Conv2D(n_filt//2, filt_size, padding='same')(p1)
    m2=LeakyReLU()(m2)
    p2 = MaxPooling2D(pool_size=(2, 2))(m2)

    m3 = Conv2D(n_filt//4, filt_size, padding='same')(p2)
    
    # アップサンプリング
    u2 = concatenate([UpSampling2D(size=(2, 2))(m3), m2])
    u2 = Conv2D(n_filt//2, filt_size, padding='same')(u2)
    u2=LeakyReLU()(u2)
    # u2 = BatchNormalization()(u2)

    u1 = concatenate([UpSampling2D(size=(2, 2))(u2), m1])
    u1 = Conv2D(n_filt, filt_size, padding='same')(u1)
    u1=LeakyReLU()(u1)

    # x=Add()([u1, skip])
    u1=concatenate([u1, skip])
    m = Conv2D(1, filt_size, padding="same")(u1)
    # m = Activation("tanh")(m)
    out1=Activation("linear", name="hologram_out")(m)
    # out1=LeakyReLU(name="hologram_out")(m)

    out2=Lambda(ld.diff_layer,name="diffract_layer")(out1)  #layer_diffract 再生計算
    
    model = Model(img_input, [out1, out2])


    return model


# U-Netのモデルを返す ホログラムで学習
def unet2(input_shape=(1, 128, 1) ):
    filt_size = 3  #フィルターのサイズ
    rate = 1
    resnet_flag = 0
    n_filt = 16  #フィルターの数
    img_input = Input(shape=input_shape)  #入力層 shape =(512,512,1)

    # ダウンサンプリング
    m1 = Conv2D(n_filt, filt_size, padding='same')(img_input) #shape (512,512,1)=>(512,512,16)
    m1 = LeakyReLU()(m1)  #活性化関数
    p1 = MaxPooling2D(pool_size=(2, 2))(m1)  #プーリング shape (512,512,16)=>(256,256,16)
    
    m2 = Conv2D(n_filt, filt_size, padding='same')(p1) #shape (256,256,16)=>(256,256,16)
    m2 = LeakyReLU()(m2)
    p2 = MaxPooling2D(pool_size=(2, 2))(m2) #shape (256,256,16)=>(128,128,16)

    m3 = Conv2D(n_filt, filt_size, padding='same')(p2) #shape (128,128,16)=>(128,128,16)
    
    # アップサンプリング
    u2 = concatenate([UpSampling2D(size=(2, 2))(m3), m2])
    u2 = Conv2D(n_filt, filt_size, padding='same')(u2)
    u2 = LeakyReLU()(u2)

    u1 = concatenate([UpSampling2D(size=(2, 2))(u2), m1])
    u1 = Conv2D(n_filt, filt_size, padding='same')(u1)
    u1 = LeakyReLU()(u1)
    
    m = Conv2D(n_filt, filt_size, padding="same")(u1)
    m = Add()([m, img_input])
    m = LeakyReLU()(m)
    m = Conv2D(1, (1, 1), padding="same")(m)
    
    m = Activation("linear")(m)
    
    model = Model(img_input, m)

    return model

# U-Netのモデルを返す(複素数画像) ホログラムで学習
def unet2_complex(input_shape=(1, 128, 1) ):
    filt_size = 3  #フィルターのサイズ
    rate = 1
    resnet_flag = 0
    n_filt = 16  #フィルターの数
    img_input = Input(shape=input_shape)  #入力層 shape =(512,512,1)

    # ダウンサンプリング
    m1 = Conv2D(n_filt, filt_size, padding='same')(img_input) #shape (512,512,1)=>(512,512,16)
    m1 = LeakyReLU()(m1)  #活性化関数
    p1 = MaxPooling2D(pool_size=(2, 2))(m1)  #プーリング shape (512,512,16)=>(256,256,16)
    
    m2 = Conv2D(n_filt, filt_size, padding='same')(p1) #shape (256,256,16)=>(256,256,16)
    m2 = LeakyReLU()(m2)
    p2 = MaxPooling2D(pool_size=(2, 2))(m2) #shape (256,256,16)=>(128,128,16)

    m3 = Conv2D(n_filt, filt_size, padding='same')(p2) #shape (128,128,16)=>(128,128,16)
    
    # アップサンプリング
    u2 = concatenate([UpSampling2D(size=(2, 2))(m3), m2])
    u2 = Conv2D(n_filt, filt_size, padding='same')(u2)
    u2 = LeakyReLU()(u2)

    u1 = concatenate([UpSampling2D(size=(2, 2))(u2), m1])
    u1 = Conv2D(n_filt, filt_size, padding='same')(u1)
    u1 = LeakyReLU()(u1)
    
    m = Conv2D(n_filt, filt_size, padding="same")(u1)
    m = Add()([m, img_input])
    m = LeakyReLU()(m)
    m = Conv2D(2, (1, 1), padding="same")(m)
    
    m = Activation("linear")(m)
    
    model = Model(img_input, m)

    return model


from keras.layers import Dense, Dropout, Activation, Flatten, Input, add
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import Model

def rescell(data, filters, kernel_size, option=False):
    strides=(1,1)
    if option:
        strides=(2,2)
    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding="same")(data)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    
    data=Conv2D(filters=int(x.shape[3]), kernel_size=(1,1), strides=strides, padding="same")(data)
    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=(1,1),padding="same")(x)
    x=BatchNormalization()(x)
    x=add([x,data])
    x=Activation('relu')(x)
    return x

# (512,512,1) → 10
def ResNet(img_rows, img_cols, img_channels, x_train):

	input=Input(shape=(img_rows,img_cols,img_channels))
	x=Conv2D(32,(7,7), padding="same", input_shape=x_train.shape[1:],activation="relu")(input)
	# x=MaxPooling2D(pool_size=(2,2))(x)

	x=rescell(x,64,(3,3))
	x=rescell(x,64,(3,3))
	x=rescell(x,64,(3,3))

	x=rescell(x,128,(3,3),True)

	x=rescell(x,128,(3,3))
	x=rescell(x,128,(3,3))
	x=rescell(x,128,(3,3))

	x=rescell(x,256,(3,3),True)

	x=rescell(x,256,(3,3))
	x=rescell(x,256,(3,3))
	x=rescell(x,256,(3,3))
	x=rescell(x,256,(3,3))
	x=rescell(x,256,(3,3))

	x=rescell(x,512,(3,3),True)

	x=rescell(x,512,(3,3))
	x=rescell(x,512,(3,3))

	x=AveragePooling2D(pool_size=(int(x.shape[1]),int(x.shape[2])),strides=(2,2))(x)

	x=Flatten()(x)
	x=Dense(units=10,kernel_initializer="he_normal",activation="softmax")(x)
	model=Model(inputs=input,outputs=[x])
	return model

def rescell2(data, filters, kernel_size):
    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=(1,1),padding="same")(data)
    # x=BatchNormalization()(x)
    x=Activation('relu')(x)
    
    data=Conv2D(filters=int(x.shape[3]), kernel_size=(1,1), strides=(1,1), padding="same")(data)
    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=(1,1),padding="same")(x)
    # x=BatchNormalization()(x)
    x=add([x,data])
    x=Activation('relu')(x)
    return x

# (512,512,1) → (512,512,1)
def ResNet2(img_rows, img_cols, img_channels, x_train):
    input=Input(shape=(img_rows,img_cols,img_channels))
    x=Conv2D(32,(7,7), padding="same", input_shape=x_train.shape[1:],activation="relu")(input)
	# x=MaxPooling2D(pool_size=(2,2))(x)
    
    x=rescell2(x,64,(3,3))
    x=rescell2(x,64,(3,3))
    x=rescell2(x,64,(3,3))
    
    x=rescell2(x,128,(3,3))
    x=rescell2(x,128,(3,3))
    x=rescell2(x,128,(3,3))

    # x=rescell(x,1,(3,3))
    # x = Add()([x, input])
    x=Conv2D(1,(3,3), padding="same", input_shape=x.shape[1:],activation="linear")(x) #reluよりlinearの方がよかった
    
    model=Model(inputs=input,outputs=[x])
    return model


# ResNet2のモデルを返す(複素数画像) ホログラムで学習
def ResNet2_complex(img_rows, img_cols, img_channels, x_train):
    input=Input(shape=(img_rows,img_cols,img_channels))
    x=Conv2D(32,(7,7), padding="same", input_shape=x_train.shape[1:],activation="relu")(input)
	# x=MaxPooling2D(pool_size=(2,2))(x)
    
    x=rescell2(x,64,(3,3))
    x=rescell2(x,64,(3,3))
    x=rescell2(x,64,(3,3))
    
    x=rescell2(x,128,(3,3))
    x=rescell2(x,128,(3,3))
    x=rescell2(x,128,(3,3))

    # x=rescell(x,1,(3,3))
    # x = Add()([x, input])
    x=Conv2D(2,(3,3), padding="same", input_shape=x.shape[1:],activation="linear")(x) #reluよりlinearの方がよかった
    
    model=Model(inputs=input,outputs=[x])
    return model

# (512,512,1) → (512,512,1)
def cnn(img_rows, img_cols, img_channels, x_train):
    input = Input(shape=(img_rows,img_cols,img_channels))
    x = Conv2D(32, (3,3), input_shape=(img_rows, img_cols, 1), activation="relu", padding="same")(input)
    x = Conv2D(64, (3,3), input_shape=x.shape[1:], activation="relu", padding="same")(x)
    x = Conv2D(1, (3,3), input_shape=x.shape[1:], activation="relu", padding="same")(x)
    model = Model(inputs=input, outputs=[x])
    return model