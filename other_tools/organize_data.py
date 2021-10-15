# rgb、d画像のファイルを移動しファイル名を変える or rgb画像のjpgからnpyを生成 or rgb画像のjpgをリサイズ
import shutil
import numpy as np
# from PIL import Image
import glob
from keras.preprocessing.image import load_img,img_to_array
from PIL import Image
# import os
import math

# rgb、d画像のファイルを移動しファイル名を変える
def remove_rename(path):
    for i in range(1500):
        num = str(i+1).zfill(4)
        shutil.copy(path+"SUNRGBD/kv1/NYUdata/NYU"+num+"/image/NYU"+num+".jpg", path+"SUNRGBD2/rgb/rgb"+num+".jpg") #rgb画像を移動&リネーム　RGB？RGBD？
        shutil.copy(path+"SUNRGBD/kv1/NYUdata/NYU"+num+"/depth_bfx/NYU"+num+".png", path+"SUNRGBD2/depth/depth"+num+".png") #depth画像を移動&リネーム　depthとdepth_bfxどっち？
        print("num:", num)

# rgb画像のjpgからnpyを生成
def generate_npy(path):
    # im = np.array(Image.open(path+'SUNRGBD2/rgb/rgb0001.jpg').convert('L')) #グレースケールにしてから読み込んだ後npyに変換
    # print(im.shape) #グレースケールなので2次元
    # pil_img = Image.fromarray(im)
    # print(pil_img.mode) #L
    # pil_img.save(path+'SUNRGBD2/rgb/rgb0001.npy')
    img_size = (256, 256) #(427, 561) #入力画像サイズ
    img_list = glob.glob(path+'SUNRGBD2/rgb_resize/rgb*.jpg') #'SUNRGBD2/rgb/rgb*.jpg' #フォルダ内のファイルパスをリスト化
    for img in img_list:
        temp_img = load_img(img, grayscale=True, target_size=(img_size)) #PIL形式で読込
        temp_img_array = img_to_array(temp_img) /255 #PIL形式からNumpy配列に変換と正規化(x,y,1)？
        file_name = img.split('\\')[1].split('.')[0]
        print(file_name, type(temp_img_array), temp_img_array.shape)
        np.save(path+'SUNRGBD2/rgb_resize/'+file_name+'.npy', temp_img_array) #'SUNRGBD2/rgb/'

# rgb画像のjpgをリサイズ
def resize_jpg(path):
    x_size = 256
    y_size = 256
    for i in range(1500):
        num = str(i+1).zfill(4)
        img = Image.open(path+'SUNRGBD2/rgb/rgb'+num+'.jpg')
        img_resize = img.resize((x_size, y_size))
        img_resize.save(path+'SUNRGBD2/rgb_resize/rgb'+num+'.jpg')

# 複素数jpg, npyの要素のmax, minを確認
def comfirm_complex_range(path):
    for num in range(1,1450):
        # #データセットからjpg画像を読み込み
        y_jpg = np.array(Image.open(path+"SUNRGBD2/hol/hol"+str(num).zfill(4)+".jpg")) #Imageで開いた後配列に変換(mode：L)
        y_npy = np.load(path+"SUNRGBD2/hol/hol"+str(num).zfill(4)+".jpg.npy")
        # print(y_jpg[0:3,0], y_npy[0:3,0])
        # print("y_jpg: ", y_jpg.shape, "y_npy: ", y_npy.shape)
        if num == 1:
            min_jpg = min(y_jpg[0,:])
            max_jpg = max(y_jpg[0,:])
            min_real = min(y_npy[0,:].real)
            max_real = max(y_npy[0,:].real)
            min_imag = min(y_npy[0,:].imag)
            max_imag = max(y_npy[0,:].imag)
        for i in range(256):
            min_jpg_t = min(y_jpg[i,:])
            max_jpg_t = max(y_jpg[i,:])
            min_real_t = min(y_npy[i,:].real)
            max_real_t = max(y_npy[i,:].real)
            min_imag_t = min(y_npy[i,:].imag)
            max_imag_t = max(y_npy[i,:].imag)
            # jpgのmax, min
            if min_jpg > min_jpg_t:
                min_jpg = min_jpg_t
            if max_jpg < max_jpg_t:
                max_jpg = max_jpg_t
            # npyのrealのmax, min
            if min_real > min_real_t:
                min_real = min_real_t
            if max_real < max_real_t:
                max_real = max_real_t
            # npyのimagのmax, min
            if min_imag > min_imag_t:
                min_imag = min_imag_t
            if max_imag < max_imag_t:
                max_imag = max_imag_t
        if num % 100 == 0:
            print("hol_num:", num)
    print("jpg min:", min_jpg, "max:", max_jpg)
    print("npyのreal min:", min_real, "max:", max_real)
    print("npyのimag min:", min_imag, "max:", max_imag)

# 複素数jpgとnpyの関係を3D-plot
def comfirm_complex_3D_plot(path):
    x, y, z = np.array([]), np.array([]), np.array([])
    for num in range(1,3,1):
        #データセットからjpg画像を読み込み
        y_jpg = np.array(Image.open(path+"SUNRGBD2/hol/hol"+str(num).zfill(4)+".jpg")) #Imageで開いた後配列に変換(mode：L)
        y_npy = np.load(path+"SUNRGBD2/hol/hol"+str(num).zfill(4)+".jpg.npy")
        
        for i in range(256):
            x = np.append(x, y_npy[i,:].real)
            y = np.append(y, y_npy[i,:].imag)
            z = np.append(z, y_jpg[i,:])

        print(y_jpg[0:3,0], y_npy[0:3,0])
        print("y_jpg: ", y_jpg.shape, "y_npy: ", y_npy.shape)
    print(x.shape, y.shape, z.shape)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.plot(x,y,z,marker=".",linestyle='None')
    plt.show()

# 複素数npyのrealとimagの関係をplot
def comfirm_complex_npy_plot(path):
    x, y = np.array([]), np.array([])
    for num in range(1,30,1):
    #データセットからjpg画像を読み込み
        y_npy = np.load(path+"SUNRGBD2/hol/hol"+str(num).zfill(4)+".jpg.npy")

        for i in range(256):
            x = np.append(x, y_npy[i,:].real)
            y = np.append(y, y_npy[i,:].imag)
        if num % 10 == 0:
            print("hol_num:", num)

    import matplotlib.pyplot as plt
    plt.plot(x,y,marker=".",linestyle='None')
    plt.show()

# 複素数jpgのtanとnpyのarctanを比較plot
def comfirm_complex_tan(path):
    x, y = np.array([]), np.array([])
    for num in range(2,4,1):
        #データセットからjpg画像を読み込み
        y_jpg = np.array(Image.open(path+"SUNRGBD2/hol/hol"+str(num).zfill(4)+".jpg")) #Imageで開いた後配列に変換(mode：L)
        y_npy = np.load(path+"SUNRGBD2/hol/hol"+str(num).zfill(4)+".jpg.npy")

        for i in range(256):
            for j in range(256):
                tan_jpg = math.tan(math.pi*y_jpg[i,j]/128-math.pi) #math.tan([rad])
                tan_npy = y_npy[i,j].imag / y_npy[i,j].real
                if abs(tan_jpg) < 25 and abs(tan_npy) < 1000: #外れ値は無視
                    x = np.append(x, tan_jpg)
                    y = np.append(y, tan_npy)
                else:
                    print(i, j)

    import matplotlib.pyplot as plt
    plt.plot(x,y,marker=".",linestyle='None')
    plt.xlabel("tan_jpg")
    plt.ylabel("tan_npy")
    plt.grid()
    plt.show()

# 複素数jpgの要素を確認
def comfirm_complex_npy_plot2(path):
    num = 3
    # nx, ny = 256, 256
    # #データセットからjpg画像を読み込み
    y_jpg = np.array(Image.open(path+"SUNRGBD2/hol/hol"+str(num).zfill(4)+".jpg")) #Imageで開いた後配列に変換(mode：L)
    y_npy = np.load(path+"SUNRGBD2/hol/hol"+str(num).zfill(4)+".jpg.npy")
    # print(y_jpg[0:3,0], y_npy[0:3,0])
    # print("y_jpg: ", y_jpg.shape, "y_npy: ", y_npy.shape)
    ans_npy = -0.03778832 / 0.13194206
    # ans_jpg = math.tan(math.radians(114*360/255))
    ans_jpg = math.tan(2*math.pi*114/255-math.pi)
    print(ans_npy, ans_jpg)

    mins = min(y_npy[0,:].imag)
    maxs = max(y_npy[0,:].imag)
    for i in range(256):
        Min = min(y_npy[i,:].imag)
        Max = max(y_npy[i,:].imag)
        if mins > Min:
            mins = Min
        if maxs < Max:
            maxs = Max
    print(mins, maxs)
    # print(y_npy[1,:].real)
    i2 = (0.13194206 + 0.12467122) / (1 + 0.12467122)
    print(i2)

    x, y = 0, 0
    for i in range(256):
        for j in range(256):
            if (y_jpg[i][j] == 128):
                x = i
                y = j
                print(x, y)
    
    print(y_npy[239, 172])

# Lambda
def Lambda_confirm():
    from keras.layers import Lambda
    import keras.backend as K
    # import numpy as np
    import tensorflow as tf

    def _multiply(args, y=2):
        x1 = args[0]
        x2 = args[1]
        return x1 * x2

    # x1=tf.convert_to_tensor([0,0,1])
    # x2=tf.convert_to_tensor([0,2,2])
    x1 = K.constant([0,0,1], dtype= tf.complex64)
    x2 = K.constant(2, dtype= tf.complex64)
    multiply = Lambda(_multiply)([x1, x2])

    print(multiply)


if __name__ == '__main__':
    path = "C:/Users/y.inoue/Desktop/Laboratory/research/dataset/"
    mode = 7 #0：rgb,d画像のファイル移動,ファイル名変更、1：rgb画像のjpgからnpyを生成、2：rgb画像のjpgをリサイズ、3：jpgを確認

    if mode == 0:
        remove_rename(path)
    elif mode == 1:
        generate_npy(path)
    elif mode == 2:
        resize_jpg(path)
    elif mode == 3:
        comfirm_complex_range(path)
    elif mode == 4:
        comfirm_complex_3D_plot(path)
    elif mode == 5:
        comfirm_complex_npy_plot(path)
    elif mode == 6:
        comfirm_complex_tan(path)
    elif mode == 7:
        Lambda_confirm()