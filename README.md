# research


＜研究の説明＞
メインの研究
・U-NetおよびResNetによるホログラムの画質改善 (低画質から高画質を生成する)
・修論とかに出した方
・関連ファイル, フォルダ：main.py, img, model

もう一つの研究
・RGB画像からホログラムを生成するやつ
・一瞬だけやって放置されてる
・関連ファイル, フォルダ：main2.py, img_sun, model_sun


＜ファイル, フォルダの説明＞
main.py (x=512, y=512)
・学習、推論などを実行
・低精度hol_fix(x,y,1) → 高精度hol_float(x,y,1)

main2.py (x=256, y=256)
・学習、推論などを実行
・rgb画像(x,y,1) → hol画像(x,y,2)

other_tools/organize_data.py
・データセットの生成とかいろいろな処理を実行できる

img, img_sun
・学習曲線などが保存されてる

model, model_sun
・学習済みモデルが保存されてる
・推論時に読み込んで使ったりする