# rgb、d画像のファイルを移動しファイル名を変える or rgb画像のjpgからnpyを生成 or rgb画像のjpgをリサイズ
import shutil
import numpy as np
# from PIL import Image
import glob
# from keras.preprocessing.image import load_img,img_to_array
from PIL import Image
# import os
import math


# jpgを再生計算(課題3参照)
def recurrent_calculation(input_file, path, pre_dir):
    import cmath
    ## inputホログラム画像の読み込み
    in1 = np.array(Image.open(path+"img"+pre_dir+"/"+input_file).convert('L')) #Imageで開いた後配列に変換(mode：L)

    in1 = in1 - np.average(in1) #平均値引いてる(npyのrealは-1～1だから？)

    NX = in1.shape[1]*2 #640*2 #1024
    NY = in1.shape[0]*2 #360*2 #1024
    lam = 633e-9 #pow(10, -9) #520 * pow(10, -9)
    delta_x = 1 * 10e-6 #pow(10, -6)
    delta_y = 1 * 10e-6 #pow(10, -6)
    delta_u = 1 / (delta_x * NX)
    delta_v = 1 / (delta_y * NY)
    # z = 0.5 #0.5, cube:1.0？, hol:0.1554
    H = np.zeros((NY,NX), dtype=np.complex) #2次元複素配列を0で初期化
    H_norm = np.zeros((NY,NX))
    g1 = np.zeros((NY,NX))
    G2 = np.zeros((NY,NX), dtype=np.complex) #2次元複素配列を0で初期化
    A_G2 = np.zeros((NY,NX))
    g2 = np.zeros((NY,NX))
    g2_norm = np.zeros((NY,NX))
    img_out = np.zeros((int(NY/2),int(NX/2)))

    ## inputホログラム画像のゼロパディング
    for ay in range(0, int(NY/2)):
        for ax in range(0, int(NX/2)):
            g1[int(NY/4)+ay][int(NX/4)+ax] = in1[ay][ax]
    # #画像を保存
    # pil_img = Image.fromarray(g1)
    # if pil_img.mode != 'RGB':
    #     pil_img = pil_img.convert('RGB') #画像をRGBに変換
    #     print("RGB")
    # pil_img.save(path+"test/A_g1.png")

    z = 0.2
    while z < 0.3:
        ## Hの計算
        for ay in range(0, NY):
            for ax in range(0, NX):
                w = math.sqrt(1.0 / lam / lam - (delta_u * (-NX / 2 + ax))**2 - (delta_v * (-NY / 2 + ay))**2)
                H[ay][ax] = cmath.exp(2j * math.pi * w * z)

        # h実部のmax, minを求める
        h_min = H[0][0].real
        h_max = H[0][0].real
        for ay in range(0, NY):
            for ax in range(0, NX):
                if h_min > H[ay][ax].real:
                    h_min = H[ay][ax].real
                if h_max < H[ay][ax].real:
                    h_max = H[ay][ax].real

        # おそらくシフト演算
        for ax in range(0, NX):
            for ay in range(0, int(NY/2)):
                if ax < NX / 2: #ay=0~511
                    temp = H[ay][ax]
                    H[ay][ax] = H[int(ay + NY / 2)][int(ax + NX / 2)] #ay+NY/2=512~
                    H[int(ay + NY / 2)][int(ax + NX / 2)] = temp
                else:
                    temp = H[ay][ax]
                    H[ay][ax] = H[int(ay + NY / 2)][int(ax - NX / 2)]
                    H[int(ay + NY / 2)][int(ax - NX / 2)] = temp
        
        # # img_outはhの実部を0～255に正規化したもの
        # for ay in range(0, NY):
        #     for ax in range(0, NX):
        #         H_norm[ay][ax] = (255 * (H[ay][ax].real - h_min)) / (h_max - h_min)

        # #画像を保存
        # pil_img = Image.fromarray(H_norm)
        # if pil_img.mode != 'RGB':
        #     pil_img = pil_img.convert('RGB') #画像をRGBに変換
        #     print("RGB")
        # pil_img.save(path+"test/H_real.jpg")

        ## FFT
        G1 = np.fft.fft2(g1) # 高速フーリエ変換(FFT)
        # G1 = np.fft.fftshift(G1)
        # print(G1)
        for ay in range(0, NY):
            for ax in range(0, NX):
                G2[ay][ax] = H[ay][ax] * G1[ay][ax] # 全て複素数
        # print(G2)
        # #画像を保存
        # for ay in range(0, NY):
        #     for ax in range(0, NX):
        #         A_G2[ay][ax] = math.sqrt((G2[ay][ax].real)**2 + (G2[ay][ax].imag)**2)
        # pil_img = Image.fromarray(A_G2)
        # if pil_img.mode != 'RGB':
        #     pil_img = pil_img.convert('RGB') #画像をRGBに変換
        #     print("RGB")
        # pil_img.save(path+"test/A_Gg2.png")
        
        # G2 = np.fft.fftshift(G2)
        # print("g1:", g1.shape, "G1:", G1.shape, "H:", H.shape, "G2:", G2.shape)
        # print("G2", G2)
        out2 = np.fft.ifft2(G2) #irfftはやべぇ！！

        # print(out2.shape)
        # print("out2", out2)
        # 逆fftしたout2の振幅をg2に格納
        for ay in range(0, NY):
            for ax in range(0, NX):
                g2[ay][ax] = math.sqrt((out2[ay][ax].real)**2 + (out2[ay][ax].imag)**2)
        # pil_img = Image.fromarray(g2)
        # if pil_img.mode != 'RGB':
        #     pil_img = pil_img.convert('RGB') #画像をRGBに変換
        #     print("RGB")
        # pil_img.save(path+"test/A_g2-1.png")

        # g2のmax, minを求める
        g2_min = g2[0][0]
        g2_max = g2[0][0]
        for ay in range(0, NY):
            for ax in range(0, NX):
                if g2_min > g2[ay][ax]:
                    g2_min = g2[ay][ax]
                if g2_max < g2[ay][ax]:
                    g2_max = g2[ay][ax]
        
        # g2を0～255に正規化し、img_outに格納
        for ay in range(0, NY):
            for ax in range(0, NX):
                g2_norm[ay][ax] = 255 * ((g2[ay][ax] - g2_min) / (g2_max - g2_min))
        # #画像を保存
        # pil_img = Image.fromarray(g2_norm)
        # if pil_img.mode != 'RGB':
        #     pil_img = pil_img.convert('RGB') #画像をRGBに変換
        #     print("RGB")
        # pil_img.save(path+"test/A_g2_norm.png")
        
        for ay in range(0, int(NY/2)):
            for ax in range(0, int(NX/2)):        
                img_out[ay][ax] = g2_norm[int(NY/4)+ay][int(NX/4)+ax]

        #画像を保存
        pil_img = Image.fromarray(img_out)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB') #画像をRGBに変換
            print("RGB")
        pil_img.save(path+"img"+pre_dir+"/rec_"+input_file.split('.')[0]+"_jpg"+".png") #A_g2
        z += 0.1

# npyを再生計算(課題3参照)
def recurrent_calculation_npy(input_file, path, pre_dir):
    import cmath
    ## inputホログラム画像の読み込み
    # in1 = np.array(Image.open(path+"img/predict/"+input_file).convert('L')) #Imageで開いた後配列に変換(mode：L)
    in1 = np.load(path+"img"+pre_dir+"/"+input_file) #"img/predict/", "img/predict_random/"
    # in1 = np.load("C:/Users/y.inoue/Desktop/Laboratory/research/object_calc/test_create_image/hol_float5.npy")

    # in1 = in1 - np.average(in1) #平均値引いてる

    NX = in1.shape[1]*2 #640*2 #1024
    NY = in1.shape[0]*2 #360*2 #1024
    lam = 633e-9 #pow(10, -9) #520 * pow(10, -9)
    delta_x = 1 * 10e-6 #pow(10, -6)
    delta_y = 1 * 10e-6 #pow(10, -6)
    delta_u = 1 / (delta_x * NX)
    delta_v = 1 / (delta_y * NY)
    # z = 0.5 #0.5, cube:1.0？, hol:0.1554
    H = np.zeros((NY,NX), dtype=np.complex) #2次元複素配列を0で初期化
    H_norm = np.zeros((NY,NX))
    g1 = np.zeros((NY,NX))
    G2 = np.zeros((NY,NX), dtype=np.complex) #2次元複素配列を0で初期化
    A_G2 = np.zeros((NY,NX))
    g2 = np.zeros((NY,NX))
    g2_norm = np.zeros((NY,NX))
    img_out = np.zeros((int(NY/2),int(NX/2)))

    ## inputホログラム画像のゼロパディング
    for ay in range(0, int(NY/2)):
        for ax in range(0, int(NX/2)):
            g1[int(NY/4)+ay][int(NX/4)+ax] = in1[ay][ax]

    z = 0.2
    while z < 0.3:
        ## Hの計算
        for ay in range(0, NY):
            for ax in range(0, NX):
                w = math.sqrt(1.0 / lam / lam - (delta_u * (-NX / 2 + ax))**2 - (delta_v * (-NY / 2 + ay))**2)
                H[ay][ax] = cmath.exp(2j * math.pi * w * z)

        # h実部のmax, minを求める
        h_min = H[0][0].real
        h_max = H[0][0].real
        for ay in range(0, NY):
            for ax in range(0, NX):
                if h_min > H[ay][ax].real:
                    h_min = H[ay][ax].real
                if h_max < H[ay][ax].real:
                    h_max = H[ay][ax].real

        # おそらくシフト演算
        for ax in range(0, NX):
            for ay in range(0, int(NY/2)):
                if ax < NX / 2: #ay=0~511
                    temp = H[ay][ax]
                    H[ay][ax] = H[int(ay + NY / 2)][int(ax + NX / 2)] #ay+NY/2=512~
                    H[int(ay + NY / 2)][int(ax + NX / 2)] = temp
                else:
                    temp = H[ay][ax]
                    H[ay][ax] = H[int(ay + NY / 2)][int(ax - NX / 2)]
                    H[int(ay + NY / 2)][int(ax - NX / 2)] = temp

        ## FFT
        G1 = np.fft.fft2(g1) # 高速フーリエ変換(FFT)
        # G1 = np.fft.fftshift(G1)
        # print(G1)
        for ay in range(0, NY):
            for ax in range(0, NX):
                G2[ay][ax] = H[ay][ax] * G1[ay][ax] # 全て複素数
        out2 = np.fft.ifft2(G2) #irfftはやべぇ！！

        # 逆fftしたout2の振幅をg2に格納
        for ay in range(0, NY):
            for ax in range(0, NX):
                g2[ay][ax] = math.sqrt((out2[ay][ax].real)**2 + (out2[ay][ax].imag)**2)

        # g2のmax, minを求める
        g2_min = g2[0][0]
        g2_max = g2[0][0]
        for ay in range(0, NY):
            for ax in range(0, NX):
                if g2_min > g2[ay][ax]:
                    g2_min = g2[ay][ax]
                if g2_max < g2[ay][ax]:
                    g2_max = g2[ay][ax]
        
        # g2を0～255に正規化し、img_outに格納
        for ay in range(0, NY):
            for ax in range(0, NX):
                g2_norm[ay][ax] = 255 * ((g2[ay][ax] - g2_min) / (g2_max - g2_min))
        
        for ay in range(0, int(NY/2)):
            for ax in range(0, int(NX/2)):        
                img_out[ay][ax] = g2_norm[int(NY/4)+ay][int(NX/4)+ax]

        #画像を保存
        pil_img = Image.fromarray(img_out)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB') #画像をRGBに変換
            print("RGB")
        pil_img.save(path+"img"+pre_dir+"/rec_"+input_file.split('.')[0]+"_npy"+".png") #predict, predict_other / A_g2
        # pil_img.save("C:/Users/y.inoue/Desktop/Laboratory/research/object_calc/test_create_image/rec_hol_float5_npy.png")
        z += 0.1

def calc_ssim(path, imgs, pre_dir):
    from skimage.metrics import structural_similarity # from skimage.measure import compare_ssim#, compare_psnr
    from sklearn.metrics import mean_squared_error
    import cv2
    
    img1 = cv2.imread(path + "img"+pre_dir+"/" + imgs[0], cv2.IMREAD_GRAYSCALE) #predict, predict_other
    img2 = cv2.imread(path + "img"+pre_dir+"/" + imgs[1], cv2.IMREAD_GRAYSCALE) #predict, predict_other
    # input_shape = (512, 512, 1)
    # img1 = load_dataset2(path+"test/hol_float%d"+".jpg", input_shape,(512,512), (0,1))
    # img2 = load_dataset2(path+"test/hol_fix%d"+".jpg", input_shape,(512,512), (0,1))
    print(imgs[1])
    print("ssim:", structural_similarity(img1, img2))
    print("mse:", mean_squared_error(img1, img2))
    print("mse:", np.average(np.square(img1-img2)))
    print(img1)
    print(img2)

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

def comfirm_complex_real(path):
    x, y = np.array([]), np.array([])
    for num in range(2,4,1):
        #データセットからjpg画像を読み込み
        y_jpg = np.array(Image.open(path+"SUNRGBD2/hol/hol"+str(num).zfill(4)+".jpg")) #Imageで開いた後配列に変換(mode：L)
        y_npy = np.load(path+"SUNRGBD2/hol/hol"+str(num).zfill(4)+".jpg.npy")
        # print(y_jpg[0][0], y_npy[0][0].real)

        max_real = 0
        min_real = 0
        for i in range(256):
            for j in range(256):
                if max_real < y_npy[0][0].real:
                    max_real = y_npy[0][0].real
                if min_real > y_npy[0][0].real:
                    min_real = y_npy[0][0].real
                # tan_npy = y_npy[i,j].imag / y_npy[i,j].real
                # if abs(tan_jpg) < 25 and abs(tan_npy) < 1000: #外れ値は無視
                #     x = np.append(x, tan_jpg)
                #     y = np.append(y, tan_npy)
                # else:
                #     print(i, j)
    print(max_real, min_real)

    # import matplotlib.pyplot as plt
    # plt.plot(x,y,marker=".",linestyle='None')
    # plt.xlabel("tan_jpg")
    # plt.ylabel("tan_npy")
    # plt.grid()
    # plt.show()

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

def renumber_img():
    import cv2
    import os
    img_path = "C:/Users/y.inoue/Desktop/Laboratory/research/hol_horn_low_accuracy_16_4_21_small/"
    img_path2 = "C:/Users/y.inoue/Desktop/Laboratory/research/hol_horn_low_accuracy_16_4_21_small_new/"
    remove_index = []

    # # ノイズファイルを除去
    # for i in range(0,508):
    #     file = glob.glob(img_path+"hol_fix"+str(i)+".jpg", recursive=False)
    #     if len(file) == 0:
    #         remove_index.append(i)

    # for i in remove_index:
    #     os.remove(img_path+"rec_float"+str(i)+".jpg")
    #     os.remove(img_path+"rec_float"+str(i)+".npy")
    #     print(i)

    # 欠番がないように詰める
    file_num = 0
    removed_num = 0
    for i in range(0,509):
        file = glob.glob(img_path+"rec_float"+str(i)+".jpg", recursive=False)
        if len(file) != 0:
            shutil.copy(img_path+"rec_float"+str(i)+".jpg", img_path2+"rec_float"+str(file_num)+".jpg") #"C:/Users/y.inoue/Desktop/hol_fix"+str(i)+".jpg"
            shutil.copy(img_path+"rec_float"+str(i)+".npy", img_path2+"rec_float"+str(file_num)+".npy")
            file_num += 1
        elif len(file) == 0:
            print(i)
            removed_num += 1
    print(file_num, removed_num)

# low holのnpyとjpgの対応 (npy：imag=0の複素数、jpg：実数)
def comfirm_npy_jpg():
    hol_path = "C:/Users/y.inoue/Desktop/Laboratory/research/hol_horn_low_accuracy_16_4_21_small/"
    num = 0
    x, y = np.array([]), np.array([])

    x_jpg = np.array(Image.open(hol_path+"hol_fix"+str(num)+".jpg")) #Imageで開いた後配列に変換(mode：L)
    x_npy = np.load(hol_path+"hol_fix"+str(num)+".npy")

    max_real = 0
    min_real = 0
    for i in range(512):
        for j in range(512):
            if max_real < x_npy[i][j].real:
                max_real = x_npy[i][j].real
            if min_real > x_npy[i][j].real:
                min_real = x_npy[i][j].real
    print(max_real, min_real)

    for i in range(512):
        for j in range(512):
            norm_npy = (x_npy[i][j].real - min_real) / (max_real - min_real) * 255
            x = np.append(x, x_npy[i][j].real)
            y = np.append(y, norm_npy)
        print(i)
    # print(x_jpg[0])
    # print(x_npy[0])
    # print(x_jpg.shape, x_npy.shape)

    import matplotlib.pyplot as plt
    plt.plot(x,y,marker=".",linestyle='None')
    plt.xlabel("npy")
    plt.ylabel("jpg")
    plt.grid()
    plt.show()

def create_dataset():
    import random
    in_path = "C:/Users/y.inoue/Desktop/Laboratory/research/hol_horn_low_accuracy_16_4_21_small/"
    out_path = "C:/Users/y.inoue/Desktop/Laboratory/research/hol_horn_low_accuracy_16_4_21_small_random/"
    num = 333
    # 学習用のデータ、ランダムな複数のholを足し合わせて新たなholを生成,保存
    # for i in range(160):
    #     n1 = random.randrange(num)
    #     n2 = random.randrange(num)
    #     x1 = np.load(in_path+"hol_fix"+str(n1)+".npy")
    #     x2 = np.load(in_path+"hol_fix"+str(n2)+".npy")
    #     y1 = np.load(in_path+"hol_float"+str(n1)+".npy")
    #     y2 = np.load(in_path+"hol_float"+str(n2)+".npy")
    #     if i < 50: #2つを足し合わせる
    #         np.save(out_path+"hol_fix"+str(i)+".npy", (x1+x2)/2) #i+num+1
    #         np.save(out_path+"hol_float"+str(i)+".npy", (y1+y2)/2) #i+num+1
    #         # print(n1, n2)
    #     elif i < 100: #3つを足し合わせる
    #         n3 = random.randrange(num)
    #         x3 = np.load(in_path+"hol_fix"+str(n3)+".npy")
    #         y3 = np.load(in_path+"hol_float"+str(n3)+".npy")
    #         np.save(out_path+"hol_fix"+str(i)+".npy", (x1+x2+x3)/3)
    #         np.save(out_path+"hol_float"+str(i)+".npy", (y1+y2+y3)/3)
    #     elif i < 160: #4つを足し合わせる
    #         n3 = random.randrange(num)
    #         n4 = random.randrange(num)
    #         x3 = np.load(in_path+"hol_fix"+str(n3)+".npy")
    #         x4 = np.load(in_path+"hol_fix"+str(n4)+".npy")
    #         y3 = np.load(in_path+"hol_float"+str(n3)+".npy")
    #         y4 = np.load(in_path+"hol_float"+str(n4)+".npy")
    #         np.save(out_path+"hol_fix"+str(i)+".npy", (x1+x2+x3+x4)/4)
    #         np.save(out_path+"hol_float"+str(i)+".npy", (y1+y2+y3+y4)/4)
    #         # print(n1, n2, n3)
    #     # np.save(out_path+"hol_fix"+str(i)+".npy", (x1+x2)/2) #i+num+1
    #     # np.save(out_path+"hol_float"+str(i)+".npy", (y1+y2)/2) #i+num+1

    # 予測用のデータ
    for i in range(200, 204):
        n1 = random.randrange(num)
        n2 = random.randrange(num)
        x1 = np.load(in_path+"hol_fix"+str(n1)+".npy")
        x2 = np.load(in_path+"hol_fix"+str(n2)+".npy")
        y1 = np.load(in_path+"hol_float"+str(n1)+".npy")
        y2 = np.load(in_path+"hol_float"+str(n2)+".npy")
        if i < 201: #2つを足し合わせる
            np.save(out_path+"hol_fix"+str(i)+".npy", (x1+x2)/2) #i+num+1
            np.save(out_path+"hol_float"+str(i)+".npy", (y1+y2)/2) #i+num+1
        elif i < 202: #3つを足し合わせる
            n3 = random.randrange(num)
            x3 = np.load(in_path+"hol_fix"+str(n3)+".npy")
            y3 = np.load(in_path+"hol_float"+str(n3)+".npy")
            np.save(out_path+"hol_fix"+str(i)+".npy", (x1+x2+x3)/3)
            np.save(out_path+"hol_float"+str(i)+".npy", (y1+y2+y3)/3)
        elif i < 203: #4つを足し合わせる
            n3 = random.randrange(num)
            n4 = random.randrange(num)
            x3 = np.load(in_path+"hol_fix"+str(n3)+".npy")
            x4 = np.load(in_path+"hol_fix"+str(n4)+".npy")
            y3 = np.load(in_path+"hol_float"+str(n3)+".npy")
            y4 = np.load(in_path+"hol_float"+str(n4)+".npy")
            np.save(out_path+"hol_fix"+str(i)+".npy", (x1+x2+x3+x4)/4)
            np.save(out_path+"hol_float"+str(i)+".npy", (y1+y2+y3+y4)/4)
        elif i < 204: #5つを足し合わせる
            n3 = random.randrange(num)
            n4 = random.randrange(num)
            n5 = random.randrange(num)
            x3 = np.load(in_path+"hol_fix"+str(n3)+".npy")
            x4 = np.load(in_path+"hol_fix"+str(n4)+".npy")
            x5 = np.load(in_path+"hol_fix"+str(n5)+".npy")
            y3 = np.load(in_path+"hol_float"+str(n3)+".npy")
            y4 = np.load(in_path+"hol_float"+str(n4)+".npy")
            y5 = np.load(in_path+"hol_float"+str(n5)+".npy")
            np.save(out_path+"hol_fix"+str(i)+".npy", (x1+x2+x3+x4+x5)/5)
            np.save(out_path+"hol_float"+str(i)+".npy", (y1+y2+y3+y4+y5)/5)
    
def create_dataset_fish():
    test_path = "C:/Users/y.inoue/Desktop/Laboratory/research/object_calc/test_create_image/"

    # ランダムな複数のholを足し合わせて新たなholを生成,保存
    x1 = np.load(test_path+"hol_fix"+str(1)+".npy")
    x2 = np.load(test_path+"hol_fix"+str(2)+".npy")
    x3 = np.load(test_path+"hol_fix"+str(3)+".npy")
    y1 = np.load(test_path+"hol_float"+str(1)+".npy")
    y2 = np.load(test_path+"hol_float"+str(2)+".npy")
    y3 = np.load(test_path+"hol_float"+str(3)+".npy")
    # RGBの3つを足し合わせる
    np.save(test_path+"hol_fix"+str(5)+".npy", (x1+x2+x3)/3)
    np.save(test_path+"hol_float"+str(5)+".npy", (y1+y2+y3)/3)


if __name__ == '__main__':
    path = "C:/Users/y.inoue/Desktop/Laboratory/research/dataset/"
    pj_path = "C:/Users/y.inoue/Desktop/Laboratory/research/tensorflow2-horn-low-accuracy-git/"
    # path = "C:/Users/y.inoue/Desktop/Laboratory/research/hol_horn_low_accuracy_16_4_21_small/"
    mode = 2 # 0:再生計算(jpg)、1:再生計算(npy)、2:SSIM, mse計算、/ 3：rgb,d画像のファイル移動,ファイル名変更、4：rgb画像のjpgからnpyを生成、5：rgb画像のjpgをリサイズ、6：jpgを確認、12:datasetからノイズ画像を除く
    dataset_type = 1 #0：オリジナル(_opj2, predict)、1：2_4 devideランダム(_2_4_divide_random, predict_random)

    if dataset_type == 0:
        pre_dir = "/predict_other" #/predict, /predict_other
    elif dataset_type == 1:
        pre_dir = "/predict_random_other" #/predict_random, /predict_random_other

    if mode == 0:
        input_file = "pre_ResNet0.jpg" #"cube140.bmp" #"pre_unet0.jpg" #rect.bmp img02.jpg hol_fix0.jpg pre_ResNet0
        recurrent_calculation(input_file, pj_path)

    elif mode == 1:
        input_file = "pre_unet0.npy" # hol_fix0, hol_float0, pre_unet0, pre_ResNet0 / pre_unet0_opj2.npy, pre_unet0_2_4_divide_random.npy // pre_unet0
        recurrent_calculation_npy(input_file, pj_path, pre_dir)

    elif mode == 2:
        # ホログラム
        # imgs = ["hol_float0.jpg", "hol_fix0.jpg"]  # hol_fix0, pre_unet0, pre_ResNet0
        # 再生像
        # imgs = ["rec_hol_float0_npy.png", "rec_pre_unet0_2_4_divide_npy.png"]  # rec_hol_fix0_npy, rec_pre_unet0_npy, rec_pre_ResNet0_npy / rec_pre_unet0_opj2_npy.png, rec_pre_unet0_2_4_divide_npy.png
        imgs = ["rec_hol_float0_npy.png", "rec_hol_fix0_npy.png"]
        calc_ssim(pj_path, imgs, pre_dir)

    elif mode == 3:
        remove_rename(path)
    elif mode == 4:
        generate_npy(path)
    elif mode == 5:
        resize_jpg(path)
    elif mode == 6:
        comfirm_complex_range(path)
    elif mode == 7:
        comfirm_complex_3D_plot(path)
    elif mode == 8:
        comfirm_complex_npy_plot(path)
    elif mode == 9:
        comfirm_complex_tan(path)
    elif mode == 10:
        comfirm_complex_real(path)
    elif mode == 11:
        Lambda_confirm()
    elif mode == 12:
        renumber_img()
    elif mode == 13:
        comfirm_npy_jpg()

    elif mode == 14: # ホログラムをランダムに足し合わせてデータセットを生成
        create_dataset()
    elif mode == 15: # rgb魚ホログラムを足し合わせる
        create_dataset_fish()