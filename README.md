# Facial-Expression-Recognition-With-Facial-Landmarks

以下為當初測試人臉辨識演算法的程式:

[`video_RetinaFace.py`](https://github.com/hung334/Facial-Expression-Recognition-With-Facial-Landmarks/blob/main/video_RetinaFace.py)

[`video_face_detection.py`](https://github.com/hung334/Facial-Expression-Recognition-With-Facial-Landmarks/blob/main/video_face_detection.py)

[`video_face_mesh.py`](https://github.com/hung334/Facial-Expression-Recognition-With-Facial-Landmarks/blob/main/video_face_mesh.py)

[`video_mtcnn.py`](https://github.com/hung334/Facial-Expression-Recognition-With-Facial-Landmarks/blob/main/video_mtcnn.py)

主要以 `video_face_mesh.py` 為主，內容包含可視化

基於 `video_face_mesh.py`修改出 `make_datasets.py` 。
`make_datasets.py` 作用為轉換數據，將每秒每幀的3D面網格的x、y、z座標以CSV檔形式做儲存。

在`./train/main.py` 測試各個回歸演算法的性能，以及儲存個模型的特徵重要性。

`./train/train.py` 5-fold 訓練回歸模型。

`./test.py`  實測輸出待測數據。

### 正面效果
![image](https://github.com/hung334/Facial-Expression-Recognition-With-Facial-Landmarks/blob/main/%E6%AD%A3%E9%9D%A2%E6%95%88%E6%9E%9C.jpg)

### 情緒識別系統
![image](https://github.com/hung334/Facial-Expression-Recognition-With-Facial-Landmarks/blob/main/%E6%83%85%E7%B7%92%E8%AD%98%E5%88%A5%E7%B3%BB%E7%B5%B1.jpg)
