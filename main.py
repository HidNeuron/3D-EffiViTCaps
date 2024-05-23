import sys

from PyQt5.QtCore import * 
from PyQt5.QtGui import * 
from PyQt5.QtWidgets import *
from thread import *
from ui import *

class MainUi (Ui_QMainWindow, QMainWindow) :

  def __init__ (self) :
    super ().__init__ ()
    self.init_ui ()
    self.setMovie ('black.gif')
    self.setSignalAndSlot ()
  
  def msg (self) :
    self.cgeCallWord('选择测试文件所在的测试夹')
    dir = QFileDialog.getExistingDirectory (None, "选择文件夹", "C:/Users/pc/Desktop/3D-EffiViTCaps")  # 起始路径
    self.tmp_dir = dir
    if self.tmp_dir :
      self.down_lef_button_2.setEnabled (True)
    self.cgeCallWord()

  def setMovie(self, gif_path):
    self.movie = QMovie(gif_path)
    self.movie.frameChanged.connect(self.updateLabel)
    self.movie.setSpeed(100)
    self.rig_label_1.setMovie(self.movie)
    self.movie.start()
    self.movie.setPaused(1)

  def updateLabel(self, frameNumber):
    # 更新标签以显示当前帧
    self.rig_label_1.setPixmap(self.movie.currentPixmap())

  def playMovie(self):
    # 开始播放GIF
    
    self.movie.start()
    self.cgeCallWord('正在播放gif中')

  def pauseMovie(self):
    # 暂停播放GIF
    self.movie.setPaused(1)
    self.cgeCallWord()

  def blockButton(self):
    self.down_lef_button_2.setEnabled(False)

  def freeButton(self):
    self.down_lef_button_2.setEnabled(True)

  def clearLabel(self, layout):
    item_list = list (range (layout.count ()))
    item_list.reverse ()
    for i in item_list :
      item = layout.itemAt (i)
      layout.removeItem (item)
      if item.widget () :
        item.widget ().deleteLater ()

  def cgeCallWord(self, word:str = '等待下一步操作'):
    self.clearLabel(self.bot_layout)
    self.bot_label_1 = QLabel('> ' + word + '...')
    self.bot_label_1.setAlignment(Qt.AlignRight)
    self.bot_label_2 = QLabel(u'计算机视觉与模式识别')
    self.bot_label_2.setAlignment(Qt.AlignLeft) 

    self.bot_layout.addWidget(self.bot_label_1, 0, 5, 1, 4)
    self.bot_layout.addWidget(self.bot_label_2, 0, 0, 1, 4)

  def start(self):
    if not self.tmp_dir : return
    # self.cgeCallWord(u'正在进行语义分割中, 请稍后')
    self.mythread = myThread(self.tmp_dir)
    self.mythread.startSignal.connect(self.blockButton)
    self.mythread.startSignal.connect(self.cgeCallWord)
    self.mythread.finishSignal.connect(self.setMovie)
    self.mythread.finishSignal.connect(self.freeButton)
    self.mythread.cgeLabelSignal.connect(self.cgeCallWord)
    self.mythread.start()

  def setSignalAndSlot(self) :
    self.down_lef_button_1.clicked.connect(self.msg)
    self.down_lef_button_2.clicked.connect(self.start)
    self.down_rig_button_3.clicked.connect(self.playMovie)
    self.down_rig_button_4.clicked.connect(self.pauseMovie)
  
def main () :
  app = QApplication (sys.argv)
  gui = MainUi ()
  styleFile = './style.qss'
  with open (styleFile, 'r') as f :
    gui.setStyleSheet (f.read ())
  gui.show ()
  ret = app.exec_ ()
  sys.exit (ret)

if __name__ == '__main__' :
  main ()
