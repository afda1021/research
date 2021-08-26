# rgb(.ppm)画像 、d(.pgm)画像を読み込む
import cv2
import glob
import numpy as np

# rgbまたはd画像を読み込む
def load_dataset_rgbd(path_train, org_shape, n_range, train_img):
    n_start = n_range[0]
    n_end = n_range[1]
    num = n_end - n_start  #画像枚数
    nx = org_shape[1]
    ny = org_shape[0]

    x = np.empty((num, ny, nx, 1)) #格納用(枚数,y,x,1)？

    # 読み込む画像のファイル名をリストに格納
    fname_list = []
    if train_img == 0:
        ftype = 'r-*.ppm'
    elif train_img == 1:
        ftype = 'd-*.pgm'
    for name in glob.glob(path_train + ftype): # "r-*.ppm" or "d-*.pgm"
        fname = name.split('\\')[1]
        fname_list.append(fname)

    # 画像を読み込んでxに格納
    cnt = 0
    for n_file in range(n_start, n_start+num):
        if train_img == 0:
            im = cv2.imread('misc_part2/home_storage_0001/' + fname_list[n_file]) #カラーで読み込み(縦, 横, 色(3))
        elif train_img == 1:
            im = cv2.imread('misc_part2/home_storage_0001/' + fname_list[n_file], cv2.IMREAD_GRAYSCALE) #グレースケール(縦, 横)
            # im = np.load('misc_part2/home_storage_0001/' + fname_list[n_file])  #ファイル読み込み(x,y)
        print(im.shape)
        for i in range(0,ny,ny):
            for j in range(0,nx,nx):
                x[cnt] = im[i:i+nx, j:j+ny].reshape(ny,nx,1)
                cnt += 1
    # cv2.imwrite('misc_part2/output.jpg', im)
    return x

if __name__ == '__main__':
    train_img = 0 #0：rgb(.ppm)画像、1：d(.pgm)画像
    nx, ny = 480, 640 #512, 512
    path_train = "C:/Users/y.inoue/Desktop/研究室関係/下馬場先生/misc_part2/home_storage_0001/"
    # 訓練データの読み込み
    x_train = load_dataset_rgbd(path_train, (ny,nx), (0,10), train_img)
    print("x_shae : ", x_train.shape)  # (枚数,x,y,1)
    