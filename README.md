## 概要
本プロジェクトは、MediaPipeを用いた顔認識と3Dモデリングを組み合わせ、顔の3D再現を行うものです。
特に、マスクなどで顔の一部が隠れている状態でも、3Dモデルを用いて顔全体の再現を目指すことを目的としています。
MediaPipeで検出した顔の特徴点に顔の3Dモデルをマッピングすることで、リアルタイムに立体的な顔の再構成が可能になります。

## 実行方法
python main.py [--texture nomask.jpg] [--draw_landmark]
python3.10 main.py
- 引数
  - texture	    使用するテクスチャ画像のファイルパスを指定します（省略可）
  - draw_landmark	ランドマークを描画する場合に指定します（省略可）

python3 create_MQO.py [model/nomask.jpg]
python3.10 /home/misa/lab/mediapipe/normal/create_MQO.py model/nomask.jpg
- 引数
  - 画像の名前は適宜変更してください

## 動作環境
OS: Microsoft Windows 11 Pro
バージョン: 10.0.26100 (ビルド 26100)
Python: 3.9.13

## 主なライブラリ
glfw==2.7.0
mediapipe==0.10.18
numpy==1.26.4
opencv-contrib-python==4.10.0.84
opencv-python==4.10.0.84
PyOpenGL==3.1.7
PyOpenGL-accelerate==3.1.7
PySimpleGUI==5.0.10

## ファイル構成
│  Application.py       # アプリの大部分の処理
│  create_MQO.py        # モデル生成
│  GLWindow.py          # glウィンドウ関連の関数
│  main.py              # メインプログラム
│  PoseEstimation.py    # カメラ姿勢推定
│  USBCamera.py         # カメラ関連の処理
│
├─data                  # 推定に使用する点のデータ
│
├─mqodata
│  │  mask.jpg
│  │  nomask.jpg        # テクスチャ画像
│  │  mesh.dat          # メッシュデータ
│  │
│  ├─landmark　         # 全特徴点保存用フォルダ
│  ├─landmark3d         # 全特徴点（正規化後）保存用フォルダ
│  ├─mesh               # メッシュデータ保存用フォルダ
│  └─model              # 生成したモデル保存用フォルダ
│      
├─mqoloader             # mqoデータ読み込みプログラム群
│
├─output　              # 出力動画像保存フォルダ
│  ├─images
│  └─videos
│
├─results　             # 結果保存フォルダ
│   culc.py
│   landmark_mask.jpg　 # ランドマーク推定結果
│
└─test_programs         # 顔認識関連のテスト用プログラム群
# lab-ishikawa-mediapipe-nomal
