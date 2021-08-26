import numpy as np

# 画像を読み込む
def load_dataset2(path, div_shape, org_shape, n_range):
    n_start = n_range[0]
    n_end = n_range[1]
    num = n_end - n_start  #画像枚数
    nx = org_shape[1]
    ny = org_shape[0]
    dx = div_shape[1]
    dy = div_shape[0]

    #print("num : ", nx//dx * ny//dy)
    x = np.empty((nx//dx * ny//dy * num, dy, dx, 1)) #格納用(枚数,y,x,1)？

    cnt = 0
    for n_file in range(n_start, n_start+num):
        fname = path % n_file  #ファイル名('～/hol_fix1.npy')
        t = np.load(fname)  #ファイル読み込み(y,x,1)

        for i in range(0,ny,dy):
            for j in range(0,nx,dx):
                x[cnt] = t[i:i+dy, j:j+dx].reshape(dy,dx,1)
                cnt += 1
    return x

# 複素数の画像を読み込む(単純化)
def load_dataset_complex(path, org_shape, n_range):
    n_start = n_range[0]
    n_end = n_range[1]
    num = n_end - n_start  #画像枚数
    nx = org_shape[1]
    ny = org_shape[0]
    x = np.empty((num, ny, nx, 2)) #格納用(枚数,2,y,x,1)？、第2引数は実部or虚部

    cnt = 0
    for n_file in range(n_start, n_start+num):
        fname = path % n_file  #ファイル名('～/hol_fix1.npy')
        t = np.load(fname)  #ファイル読み込み(y,x,1)
        x[cnt,:,:,0] = t.real #虚部 [:, :, np.newaxis] .reshape(ny,nx,1)
        x[cnt,:,:,1] = t.imag #実部
        cnt += 1

    return x

# (1枚,x,y,1)を(x,y)の画像に変換
def as_img(data, div_shape, org_shape):
    dx = div_shape[1]
    dy = div_shape[0]
    ox = org_shape[1]
    oy = org_shape[0]
    img = np.empty((oy, ox, 1))
    # print(img.shape)

    cnt = 0
    for i in range(0,oy,dy):
        for j in range(0,ox,dx):
            # print(j)
            # print(data[cnt].shape)
            img[i:i+dy, j:j+dx, ]=data[cnt]
            # print(i, i+dy, j, j+dx)
            cnt+=1

    img=img.reshape(oy,ox)
    img=img.astype(np.complex64)
    return img