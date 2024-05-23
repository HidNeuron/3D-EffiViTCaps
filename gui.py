import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PyQt5.QtGui import QMovie

class GifPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initMovie()

    def initUI(self):
        # 创建两个按钮，分别用于播放和暂停GIF
        self.playButton = QPushButton('播放')
        self.pauseButton = QPushButton('暂停')
        
        # 连接按钮的clicked信号到相应的槽函数
        self.playButton.clicked.connect(self.playMovie)
        self.pauseButton.clicked.connect(self.pauseMovie)
        
        # 创建一个垂直布局，并添加按钮和标签
        layout = QVBoxLayout()
        layout.addWidget(self.playButton)
        layout.addWidget(self.pauseButton)
        self.setLayout(layout)
        
        # 创建一个标签用于显示GIF
        self.label = QLabel()
        layout.addWidget(self.label)

    def initMovie(self):
        # 创建QMovie对象并设置GIF文件路径
        self.movie = QMovie('array.gif')  # 替换为你的GIF文件路径
        self.movie.frameChanged.connect(self.updateLabel)
        self.movie.setSpeed (100)  # 设置播放速度

    def updateLabel(self, frameNumber):
        # 更新标签以显示当前帧
        self.label.setPixmap(self.movie.currentPixmap())

    def playMovie(self):
        # 开始播放GIF
        # if not self.movie.Running ():
        self.movie.start()

    def pauseMovie(self):
        # 暂停播放GIF
        # if self.movie.Running ():
        self.movie.setPaused(1)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GifPlayer()
    ex.show()
    sys.exit(app.exec_())