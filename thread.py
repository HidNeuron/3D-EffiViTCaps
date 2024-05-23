from PyQt5.QtCore import QThread, pyqtSignal
import os, shutil
from time import sleep

from PyQt5.QtCore import * 
from PyQt5.QtGui import * 
from PyQt5.QtWidgets import *


from inference import inference
from utils import show_nii_gif, gif_scale

class myThread (QThread) :
  '''
    对医学图像进行推理的线程
  '''

  startSignal = pyqtSignal (str)
  finishSignal = pyqtSignal (str)
  cgeLabelSignal = pyqtSignal (str)

  def __init__(self, src_dir, parent = None) :
    super (myThread, self).__init__ (parent)
    self.tar_dir = './dataset/iseg/domainA_val'
    self.src_dir = src_dir
    
  def run (self) :
    self.startSignal.emit (u'正在进行语义分割中, 请稍后')
    shutil.rmtree(self.tar_dir)  
    os.mkdir(self.tar_dir)
    for file in os.listdir (self.src_dir) :
      file_path = os.path.join (self.src_dir, file)
      suffix =  ['label.hdr', 'label.img', 'T1.hdr', 
                 'T1.img', 'T2.hdr', 'T2.img']
      for suf in suffix :
        if suf in file : 
          save_path = os.path.join(self.tar_dir, 'tar-' + suf) 
          shutil.copyfile(file_path, save_path)
          break
     
    inference()
    show_nii_gif('output_dir/dataset/iseg/domainA_val/tar-label/tar-label_modified-ucaps_prediction.nii.gz')
    gif_scale('result.gif')

    self.finishSignal.emit ('result.gif')
    self.cgeLabelSignal.emit ('等待下一步操作')
