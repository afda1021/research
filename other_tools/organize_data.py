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

# 複素数jpgの要素を確認
def comfirm_complex_jpg(path):
    num = 3
    nx, ny = 256, 256
    #データセットからjpg画像を読み込み
    y_jpg = np.array(Image.open(path+"SUNRGBD2/hol/hol"+str(num).zfill(4)+".jpg")) #Imageで開いた後配列に変換(mode：L)
    y_npy = np.load(path+"SUNRGBD2/hol/hol"+str(num).zfill(4)+".jpg.npy")
    print(y_jpg[0:3,0], y_npy[0:3,0])
    print("y_jpg: ", y_jpg.shape, "y_npy: ", y_npy.shape)
    # ans_npy = -0.03778832 / 0.13194206
    # # ans_jpg = math.tan(math.radians(114*360/255))
    # ans_jpg = math.tan(2*math.pi*114/255-math.pi)
    # print(ans_npy, ans_jpg)

    # mins = min(y_npy[0,:].imag)
    # maxs = max(y_npy[0,:].imag)
    # for i in range(256):
    #     Min = min(y_npy[i,:].imag)
    #     Max = max(y_npy[i,:].imag)
    #     if mins > Min:
    #         mins = Min
    #     if maxs < Max:
    #         maxs = Max
    # print(mins, maxs)
    # # print(y_npy[1,:].real)
    # i2 = (0.13194206 + 0.12467122) / (1 + 0.12467122)
    # print(i2)

    # x, y = 0, 0
    # for i in range(256):
    #     for j in range(256):
    #         if (y_jpg[i][j] == 128):
    #             x = i
    #             y = j
    #             print(x, y)
    
    # print(y_npy[239, 172])

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    x, y, z = np.array([]), np.array([]), np.array([])
    for i in range(256):
        x = np.append(x, y_npy[i,:].real)
        y = np.append(y, y_npy[i,:].imag)
        z = np.append(z, y_jpg[i,:])

    print(x.shape, y.shape, z.shape)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.plot(x,y,z,marker=".",linestyle='None')

    plt.show()

if __name__ == '__main__':
    path = "C:/Users/y.inoue/Desktop/Laboratory/research/dataset/"
    mode = 3 #0：rgb,d画像のファイル移動,ファイル名変更、1：rgb画像のjpgからnpyを生成、2：rgb画像のjpgをリサイズ、3：jpgを確認

    if mode == 0:
        remove_rename(path)
    elif mode == 1:
        generate_npy(path)
    elif mode == 2:
        resize_jpg(path)
    elif mode == 3:
        comfirm_complex_jpg(path)