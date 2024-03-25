# coding:utf-8
# TODO 添加一个图形化界面
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
from net import Inception10
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import pandas as pd

names = dir_list = os.listdir("data")
df = pd.read_csv('label.csv')
df = df.fillna(1)




class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('imgs/面性铅笔.png'))
        self.setWindowTitle('狗狗分类')
        # 加载网络
        self.resize(800, 600)
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        font = QFont('楷体', 15)
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        img_title = QLabel("测试样本")
        img_title.setFont(font)
        img_title.setAlignment(Qt.AlignCenter)
        self.img_label = QLabel()
        self.predict_img_path = "test/1.jpg"
        img_init = cv2.imread(self.predict_img_path)
        img_init = cv2.resize(img_init, (128, 128))
        cv2.imwrite('imgs/target.png', img_init)
        self.img_label.setPixmap(QPixmap('imgs/target.png'))
        left_layout.addWidget(img_title)
        left_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        left_widget.setLayout(left_layout)

        right_widget = QWidget()
        right_layout = QVBoxLayout()
        btn_change = QPushButton(" 上传图像 ")
        btn_change.clicked.connect(self.change_img)
        btn_change.setFont(font)
        btn_predict = QPushButton(" 识别狗狗类别")
        btn_predict.setFont(font)
        btn_predict.clicked.connect(self.predict_img)

        label_result = QLabel(' 识 别 结 果 ')
        self.result = QLabel("待识别")
        label_result.setFont(QFont('楷体', 16))
        self.result.setFont(QFont('楷体', 24))
        right_layout.addStretch()
        right_layout.addWidget(label_result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(btn_change)
        right_layout.addWidget(btn_predict)
        right_layout.addStretch()
        right_widget.setLayout(right_layout)

        # 关于页面
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('训练的损失和精度')
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        # about_img.setPixmap(QPixmap('imgs/logoxx.png'))
        about_img.setPixmap(QPixmap('imgs/wxs.jpg'))
        about_img.setAlignment(Qt.AlignCenter)
        label_super = QLabel()
        label_super.setFont(QFont('楷体', 12))
        label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        # git_img = QMovie('images/')
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_widget.setLayout(main_layout)
        self.addTab(main_widget, '主页面')
        self.addTab(about_widget, '训练结果展示')
        self.setTabIcon(0, QIcon('imgs/面性计算器.png'))
        self.setTabIcon(1, QIcon('imgs/面性本子vg.png'))

    def change_img(self):
        openfile_name = QFileDialog.getOpenFileName(self, '选择文件', '', 'Image files(*.jpg , *.png, *.jpeg)')
        print(openfile_name)
        img_name = openfile_name[0]
        if img_name == '':
            pass
        else:
            self.predict_img_path = img_name
            img_init = cv2.imread(self.predict_img_path)
            img_init = cv2.resize(img_init, (400, 400))
            cv2.imwrite('imgs/target.png', img_init)
            self.img_label.setPixmap(QPixmap('imgs/target.png'))

    def predict_img(self):
        # 预测图片
        # 开始预测
        # img = Image.open()
        imgData = []
        img = Image.open(self.predict_img_path)
        img = img.convert("RGB")
        img = img.resize((64, 64), Image.ANTIALIAS)
        img = np.array(img)
        img = img / 255.0
        imgData.append(img)
        imgData = np.array(imgData)
        model = Inception10(num_blocks=6, num_classes=18)
        model.load_weights("./checkpoint/Inception10.ckpt")
        result = model.predict(imgData)
        index = tf.argmax(result, axis=1)
        result = str(names[index[0]])
        filtered_values = df[df['原文件名名'] == result][["犬类中文名", "是否攻击性"]].values

        show_text = "犬名："
        for index, ele in enumerate(filtered_values[0]):
            if index == 0:
                show_text = show_text + ele
            if index == 1:
                if ele != 1:
                    show_text = show_text + ";是否攻击性:" + ele
        self.result.setText(show_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    x = MainWindow()
    x.show()
    sys.exit(app.exec_())

# TODO: Add batch classification support
# TODO: Add model switching in GUI
