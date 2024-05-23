from PyQt5.QtCore import * 
from PyQt5.QtGui import * 
from PyQt5.QtWidgets import *
# from thread import *
import qtawesome 

class QSoundLabel (QLabel) :
  def __init__ (self, text) :
    super ().__init__ ()
    self.setText (text)

class QMusicalNoteLabel (QLabel) :
  def __init__ (self, img_path) :
    super ().__init__ ()
    img = QPixmap (img_path)
    self.setPixmap (img)

class QTitleLabel (QLabel) :
  def __init__ (self, text) :
    super ().__init__ ()
    self.setText (text)

class Ui_QMainWindow (QMainWindow) :

  def __init__ (self) :
    super ().__init__ ()
    self.init_ui ()
  
  def fixRowHeight(self, layout, row, height):
    for i in range(layout.count()):
        item = layout.itemAt(i)
        r, _, _, _ = layout.getItemPosition(i)
        if item.widget() is not None and r == row:
            item.widget().setFixedHeight(height)

  def init_ui (self) :
    self.setFixedSize (960,700)
    self.setWindowTitle ('3D-EffiViTCaps')
    self.setWindowIcon (QIcon ('./figures/icon.jpg'))

    self.main_widget = QWidget () # 创建窗口主部件
    self.main_layout = QGridLayout () # 创建主部件的网格布局
    self.main_widget.setLayout (self.main_layout) # 设置窗口主部件布局为网格布局

    ###############################################
    ##                                           ##
    ##                 title                     ##
    ##                                           ##
    ###############################################
    ##                  ##                       ##
    ##   group name     ##          gif          ##
    ##   stu1 name      ##                       ##
    ##   sut2 name      ##                       ##
    ###############################################
    ##                  ##                       ##
    ##  file    segment ##   play        stop    ##
    ##                  ##                       ##
    ###############################################
    ##                             call-word     ##
    ###############################################

    ###############################################
    # 上侧部件设计 #################################
    self.up_widget = QWidget ()
    self.up_layout = QGridLayout ()
    self.up_widget.setLayout (self.up_layout)

    self.up_label_1 = QTitleLabel ('3D-EffiViTCaps')
    self.up_label_1.setAlignment (Qt.AlignCenter)
    self.up_hline_1 = QFrame ()
    self.up_hline_1.setFrameShape (QFrame.HLine)
    self.up_hline_1.setFrameShadow (QFrame.Raised)

    self.up_layout.addWidget (self.up_label_1, 0, 0, 1, 10)
    self.up_layout.addWidget (self.up_hline_1, 1, 0, 1, 10)

    ###############################################
    # 中左侧部件设计 #################################
    self.lef_widget = QWidget ()
    self.lef_layout = QGridLayout ()
    self.lef_widget.setLayout (self.lef_layout)

    self.lef_label_1 = QLabel (u'小组成员:')
    self.lef_label_1.setAlignment (Qt.AlignCenter)
    self.lef_label_2 = QLabel (u'甘东伟')
    self.lef_label_2.setAlignment (Qt.AlignCenter)
    self.lef_label_3 = QLabel (u'常铭')
    self.lef_label_3.setAlignment (Qt.AlignCenter)
    self.lef_label_4 = QLabel (u'谭杰元')
    self.lef_label_4.setAlignment (Qt.AlignCenter)
    self.lef_label_5 = QLabel (u'黄奕鸿')
    self.lef_label_5.setAlignment (Qt.AlignCenter)
    self.lef_label_6 = QLabel (u'文件夹内文件格式应为: *.hdr + *.img')
    self.lef_label_6.setAlignment (Qt.AlignCenter)

    self.lef_layout.addWidget (self.lef_label_1, 0, 0, 1, 4)
    self.lef_layout.addWidget (self.lef_label_2, 1, 0, 1, 4)
    self.lef_layout.addWidget (self.lef_label_3, 2, 0, 1, 4)
    self.lef_layout.addWidget (self.lef_label_4, 3, 0, 1, 4)
    self.lef_layout.addWidget (self.lef_label_5, 4, 0, 1, 4)
    self.lef_layout.addWidget (self.lef_label_6, 5, 0, 1, 4)

    ###############################################
    # 中右部分部件设计 #############################
    self.rig_widget = QWidget ()
    self.rig_layout = QGridLayout ()
    self.rig_widget.setLayout (self.rig_layout)
    
    self.rig_label_1 = QLabel ('')
    self.rig_label_1.setAlignment (Qt.AlignCenter)
    self.rig_layout.addWidget (self.rig_label_1, 0, 0, 5, 6)

    ###############################################
    # 下左侧部件设计 #############################
    self.down_lef_widget = QWidget ()
    self.down_lef_layout = QGridLayout ()
    self.down_lef_widget.setLayout (self.down_lef_layout)
        
    self.down_lef_button_1 = QPushButton (text = u'选择文件夹')
    self.down_lef_button_1.setFixedHeight (35)
    self.down_lef_button_2 = QPushButton (text = u'语义分割')
    self.down_lef_button_2.setFixedHeight (35)
    self.down_lef_button_2.setEnabled (False)

    self.down_lef_layout.addWidget (self.down_lef_button_1, 0, 0, 2, 2)
    self.down_lef_layout.addWidget (self.down_lef_button_2, 0, 2, 2, 2)
    
    ###############################################
    # 下右侧部件设计 ###############################
    self.down_rig_widget = QWidget()
    self.down_rig_layout = QGridLayout()
    self.down_rig_widget.setLayout(self.down_rig_layout)

    self.down_rig_button_3 = QPushButton (qtawesome.icon ('fa.play-circle'), u'播放')
    self.down_rig_button_3.setFixedHeight (35)
    self.down_rig_button_4 = QPushButton (qtawesome.icon ('ei.repeat'), u'暂停')
    self.down_rig_button_4.setFixedHeight (35)

    self.down_rig_layout.addWidget (self.down_rig_button_3, 0, 0, 2, 2)
    self.down_rig_layout.addWidget (self.down_rig_button_4, 0, 2, 2, 2)

    ###############################################
    # 底部部件设计 ###############################
    self.bot_widget = QWidget()
    self.bot_layout = QGridLayout()
    self.bot_widget.setLayout(self.bot_layout)

    self.bot_label_1 = QLabel(u'> 等待下一步操作...')
    self.bot_label_1.setAlignment(Qt.AlignRight)
    self.bot_label_2 = QLabel(u'计算机视觉与模式识别')
    self.bot_label_2.setAlignment(Qt.AlignLeft) 
    
    self.bot_layout.addWidget(self.bot_label_1, 0, 5, 1, 4)
    self.bot_layout.addWidget(self.bot_label_2, 0, 0, 1, 4)

    ###############################################
    # 整体布局设计 #################################
    self.main_layout.addWidget (self.up_widget, 0, 0, 1, 9)
    self.main_layout.addWidget (self.lef_widget, 3, 0, 5, 3)
    self.main_layout.addWidget (self.rig_widget, 3, 3, 5, 6)
    self.main_layout.addWidget (self.down_lef_widget, 8, 0, 1, 3)
    self.main_layout.addWidget (self.down_rig_widget, 8, 3, 1, 6)
    self.main_layout.addWidget (self.bot_widget, 9, 0, 1, 9)

    # self.up_widget.setStyleSheet('''QWidget{background-color:#66CCFF;}''')
    # self.lef_widget.setStyleSheet('''QWidget{background-color:#66ffcc;}''')
    # self.rig_widget.setStyleSheet('''QWidget{background-color:#99ffff;}''')
    # self.down_lef_widget.setStyleSheet('''QWidget{background-color:#ee0000;}''')
    # self.down_rig_widget.setStyleSheet('''QWidget{background-color:#66cdef;}''')
    self.bot_widget.setStyleSheet('''QWidget{background-color:#D3D3D3;}''')

    self.fixRowHeight (self.main_layout, 9, 25)
    self.bot_layout.setContentsMargins (2, 2, 2, 2)
    self.main_layout.setContentsMargins (0, 0, 0, 0)

    self.setCentralWidget (self.main_widget)
