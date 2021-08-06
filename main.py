## dataset：hol_horn

#----------------------GPUの上限とか設定する何か？----------------------------
#https://qiita.com/masudam/items/c229e3c75763e823eed5
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
#---------------------------------------------------------------------------

import os
import keras
from keras import optimizers
import model
from keras import backend as K
import utility_depth_predict as util
from keras.callbacks import ModelCheckpoint,CSVLogger,ReduceLROnPlateau
import cv2
from PIL import Image
# プロット用
import matplotlib.pyplot as plt
import numpy as np
# 再生計算用
from keras.layers import Lambda
import layer_diffract as ld

# 損失の履歴をプロット
def plot_loss(history, model_name):
    #グラフ表示
    #plt.figure(figsize=(12, 10))
    #plt.rcParams['font.family'] = 'Times New Roman'
    #plt.rcParams['font.size'] = 25  # 全体のフォント
    #損失の履歴をプロット
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    plt.plot(range(1,epochs+1), loss, linestyle = "solid", label='train loss') #marker='.'
    plt.plot(range(1,epochs+1), val_loss, label='valid loss')
    plt.xticks(np.arange(1, epochs+1, 1)) #x軸は1刻み
    if model_name == "unet":
        plt.title('U-Net')
    elif model_name == "ResNet":
        plt.title('ResNet')
    plt.legend(loc='upper right') #fontsize=12
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig("./img/loss_"+model_name+".png") #plt.savefig("./img/graph.eps",dpi=600)
    plt.show()

# 損失の正答率をプロット
def plot_accuracy(history, model_name):
    #グラフ表示
    #plt.figure(figsize=(12, 10))
    #plt.rcParams['font.family'] = 'Times New Roman'
    #plt.rcParams['font.size'] = 25  # 全体のフォント
    #損失の履歴をプロット
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    plt.plot(range(1,epochs+1), accuracy, linestyle = "solid", label='train accuracy') #marker='.'
    plt.plot(range(1,epochs+1), val_accuracy, label='valid accuracy')
    plt.xticks(np.arange(1, epochs+1, 1)) #x軸は1刻み
    if  model_name == "unet":
        plt.title('U-Net')
    elif  model_name == "ResNet":
        plt.title('ResNet')
    plt.legend(loc='upper right') #fontsize=12
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig("./img/accuracy_"+model_name+".png") #plt.savefig("./img/graph.eps",dpi=600)
    plt.show()

# 損失関数
def loss_mse_l1(y_true, y_pred):
    weight = 1e-2
    loss = K.mean(K.square(y_pred - y_true) + K.abs(y_pred) * weight, axis=-1) #K.squareは各要素を2乗　y_pred-y_trueは要素ごとに演算
    return loss


if __name__ == '__main__':
    training = 0 #0：学習、1：テスト(jpg)、2： テスト(bmp)？、3：テスト(jpg)？、4：再生計算？
    model_type = 0 #0：U-Net、1：ResNet

    batch_size = 10
    epochs = 10

    modelDirectory = os.getcwd()  #カレントディレクトリを取得

    nx, ny = 128, 128 #512, 512
    d_nx, d_ny = 128, 128 #512, 512
    input_shape = (d_ny, d_nx, 1)

    path_train = "C:/Users/y.inoue/Desktop/Laboratory/research/hol_horn_low_accuracy_16_4_21_small/"
    
    #loss_func = "mse"  #損失関数
    lr=1e-4  #lerning rate
    adam = optimizers.Adam(lr=lr)

    if model_type == 0:
        net = model.unet2(input_shape)  #U-Netのモデル
        model_name = "unet"
    elif model_type == 1:
        x_train = util.load_dataset2(path_train+"hol_fix%d"+".npy", input_shape,(ny,nx), (0,1))
        net = model.ResNet2(d_nx, d_ny, 1, x_train)  #ResNetのモデル
        # net = model.cnn(d_nx, d_ny, 1, x_train)  #cnnのモデル
        model_name = "ResNet"
    
    net.compile(loss=loss_mse_l1, optimizer=adam, metrics=['accuracy'])
    net.summary()  #モデル形状を表示

    if training == 0: #学習、モデルの保存、学習曲線の表示と保存
        ext = ".npy"
        # 訓練データの読み込み
        x_train = util.load_dataset2(path_train+"hol_fix%d"+ext, input_shape,(ny,nx), (0,140)) #(0,496)
        print("x_shae : ", x_train.shape)  # (枚数,x,y,1)
        y_train = util.load_dataset2(path_train+"hol_float%d"+ext, input_shape,(ny,nx), (0,140))
        print("y_shae : ", y_train.shape)
        # 評価データの読み込み
        x_val = util.load_dataset2(path_train+"hol_fix%d"+ext, input_shape,(ny,nx), (140,150)) #(496,506)
        print("x_shae : ", x_val.shape)
        y_val = util.load_dataset2(path_train+"hol_float%d"+ext, input_shape,(ny,nx), (140,150))
        print("y_shae : ", y_val.shape)
        # テストデータの読み込み(予測用)
        x_test = util.load_dataset2(path_train+"hol_fix%d"+ext, input_shape,(ny,nx), (157,158)) #(506,507)
        print("x_shae : ", x_test.shape)
        y_test = util.load_dataset2(path_train+"hol_float%d"+ext, input_shape,(ny,nx), (157,158))

        x_train = x_train.astype('float32')
        y_train = y_train.astype('float32')
        x_val = x_val.astype('float32')
        y_val = y_val.astype('float32')
        x_test = x_test.astype('float32')

        ### callbacks 
        cp_cb = keras.callbacks.ModelCheckpoint( #model~.hdf5にモデルを保存
                filepath = "model/model_" + model_name + ".hdf5",
                monitor='val_loss', #監視する値
                verbose=1, save_best_only=True, mode='auto')
        csv_logger = CSVLogger('model.log') #model.logにログを書く
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, #patience値の間に更新がなかったら学習率をfactor倍する
                                        patience=10, min_lr=lr*0.001, verbose=1)

        # 学習
        history = net.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=True, validation_data=(x_val, y_val) ,callbacks=[reduce_lr, cp_cb, csv_logger])
        # 予測
        pre = net.predict(x_test, verbose=0)
        print("pre shape : ", pre.shape)
        nx = x_test.shape[2]
        ny = x_test.shape[1]
        x_test = util.as_img(x_test,input_shape,(d_ny,d_nx))  # 入力画像 (x,y)
        y_test = util.as_img(y_test,input_shape,(d_ny,d_nx))  # 正解画像
        pre = util.as_img(pre,input_shape,(d_ny,d_nx))  # 予測画像
        print(x_test.shape)

        # 損失をプロット, 保存
        plot_loss(history, model_name)
        # 正答率をプロット, 保存
        plot_accuracy(history, model_name)
    
    elif training == 1: #入力画像(jpg)から予測画像(jpg)を生成し保存
        ext = ".jpg"
        num = 247
        #データセットからjpg画像を読み込み
        x_test = np.array(Image.open(path_train+"/hol_fix"+str(num)+ext).resize((nx, ny))) #Imageで開いた後配列に変換(mode：L)
        print("x_shape : ", x_test.shape)
        x_test = x_test[np.newaxis, ...]
        x_test = x_test.reshape(1,nx,ny,1)
        print("x_shape : ", x_test.shape)
        #モデルを読み込み
        fname_weight = modelDirectory + "/model_" + model_name + ".hdf5"
        #fname_weight = modelDirectory + "/model.hdf5"
        net.load_weights(fname_weight)
        #予測
        pre = net.predict(x_test, verbose=0)
        print("pre_shape : ", pre.shape)
        pre = pre[0]
        pre = pre[:,:,0]
        print("pre_shape : ", pre.shape)
        #画像を保存
        pil_img = Image.fromarray(pre)
        if pil_img.mode != 'L': # #L：8ビットピクセル画像。黒と白  RGB
            pil_img = pil_img.convert('L') #画像をLに変換  RGB
            print("L")
        pil_img.save(modelDirectory+"/img/predict/pre_"+model_name+str(num)+ext)

    elif training == 2: #入力画像(npy)から予測画像(bmp)を生成しようとした、未完成
        ext = ".npy"
        num = 0
        #データセットからnpy画像を読み込み
        x_test = Image.open(path_train+"/hol_fix"+str(num)+ext)
        print("type",type(x_test))
        print("mode",x_test.mode)
        x_test = np.array()
        print("x_shape : ", x_test.shape)
        x_test = x_test.astype('float32')
        # x_test /= x_test.max()
        #モデルを読み込み
        fname_weight = modelDirectory + "/model_" + model_name + ".hdf5"
        net.load_weights(fname_weight)
        #予測
        pre = net.predict(x_test, verbose=0)
        pre = util.as_img(pre,input_shape,(d_ny,d_nx)) #1024

        # x_test = util.as_img(x_test,input_shape,(d_ny,d_nx))
        # x_test = x_test.reshape(512,512,1)
        x_test = x_test[0]
        predict_path = "./img/predict/pre.bmp"
        cv2.imwrite(predict_path, x_test)

    elif training == 3: #予測画像を再生計算して保存
        ext = ".jpg"
        num = 0
        #データセットからjpg画像を読み込み
        x_test = np.array(Image.open(path_train+"/hol_fix"+str(num)+ext))
        print("x_shape : ", x_test.shape)
        x_test = x_test[np.newaxis, ...]
        x_test = x_test.reshape(1,512,512,1)
        print("x_shape : ", x_test.shape)
        #モデルを読み込み
        fname_weight = modelDirectory + "/model_" + model_name + ".hdf5"
        net.load_weights(fname_weight)
        #予測
        pre = net.predict(x_test, verbose=0)
        #再生計算
        pre = K.constant(pre) #テンソルに変換
        pre = Lambda(ld.diff_layer,name="diffract_layer")(pre)  #layer_diffract 再生計算
        pre = K.eval(pre) #numpy配列に戻す
        
        print("pre_shape : ", pre.shape)
        pre = pre[0]
        pre = pre[:,:,0]
        print("pre_shape : ", pre.shape)
        #画像を保存
        pil_img = Image.fromarray(pre)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB') #画像をRGBに変換
            print("RGB")
        pil_img.save(modelDirectory+"/img/predict/rec_pre"+str(num)+ext)
    
    elif training == 4: #入力画像を再生計算して保存
        ext = ".jpg"
        num = 0
        #画像を読み込み
        x_test = np.array(Image.open(path_train+"/hol_fix"+str(num)+ext))
        print("x_shape : ", x_test.shape)
        x_test = x_test[np.newaxis, ...]
        x_test = x_test.reshape(1,512,512,1)
        print("x_shape : ", x_test.shape)
        #再生計算 Fで出力されるからそこをなんとかしないと！
        x_test = K.constant(x_test) #テンソルに変換
        x_test = Lambda(ld.diff_layer,name="diffract_layer")(x_test)  #layer_diffract 再生計算
        x_test = K.eval(x_test) #numpy配列に戻す
        print("x_test : ", x_test.shape)
        x_test = x_test[0]
        x_test = x_test[:,:,0]
        print("x_test : ", x_test.shape)
        #画像を保存
        pil_img = Image.fromarray(x_test) #配列から画像を生成、引数(データ, mode)
        print("mode", pil_img.mode) #FじゃなくてLじゃないとまずいよ！
        if pil_img.mode != 'L':
            pil_img = pil_img.convert('L') #画像をRGBに変換
            print("L")
        pil_img.save(modelDirectory+"/img/rex_x_test"+str(num)+ext)

    elif training == 5: #実験
        x_test = Image.open(path_train+"/hol_fix0.jpg")
        x_test = Image.open(modelDirectory+"/img/predict/pre0.jpg")
        print("type",type(x_test))
        print("mode",x_test.mode)
        x_test = np.array(x_test)
        print("x_shape : ", x_test.shape)
        # x_test = x_test.reshape(500,500)
        pil_img = Image.fromarray(x_test)
        print(pil_img.mode)
        # if pil_img.mode != 'RGB':
        #     pil_img = pil_img.convert('RGB') #画像をRGBに変換
        #     print("RGB")
        pil_img.save(modelDirectory+"/img/sample.png")