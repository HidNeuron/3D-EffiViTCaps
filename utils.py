import cv2
import nibabel as nib
import numpy as np

from PIL import Image
from PIL import ImageSequence

def show_nii_gif(nii_file, save_path="result.gif", show_frame=True):
  """
  @Brife:
      将 shape=(arr_len, h, w) 的 nii 文件转化为 gif
  @Param:
      nii_file   : nii 文件的路径
      show_frame : 是否展示这是第几帧, 默认 True
      save_path  : 默认为当前路径的 array.gif 文件
  """
  
  # 调包加载 nii 文件
  img=nib.load(nii_file)
  # 转化成 numpy.ndarray 的格式
  img_arr = img.get_fdata()
  img_arr = np.squeeze(img_arr)
  img_arr = img_arr.transpose ([2, 0, 1])

  cnt = img_arr.shape[0]
  idx_l, idx_r = cnt // 5, cnt - cnt // 5
  img_arr = img_arr[idx_l : idx_r]

  # 找到最大最小值, 方便之后归一化
  img_max, img_min = img_arr.max(), img_arr.min()
  
  # 归一化
  img_arr = (img_arr - img_min) / (img_max - img_min) * 255
  img_arr = img_arr.astype(np.uint8)
  
  # 将单通道的灰度图转化为RGB三通道的灰色图, (不转化这一步, 没法写字)
  img_RGB_list = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_arr]
  
  assert 3==len(img_arr.shape) # 如果是别的shape的, 那处理不了
  
  arr_len, h, w = img_arr.shape
  
  if show_frame:
    # 在每一帧上打上这是第几帧
    for i in range(arr_len):
        
      img_RGB_list[i] = cv2.putText(img_RGB_list[i], 
                                '{:>03d}/{:>03d}'.format(i+1, arr_len),
                                (10, 20),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.5,
                                (255, 255, 255),
                                1)
  
  # 将所有的 ndarray 转化为 PIL.Image 格式
  imgs = [Image.fromarray(img) for img in img_RGB_list]
  
  # 保存
  # duration is the number of milliseconds between frames; this is 40 frames per second
  imgs[0].save(save_path, save_all=True, append_images=imgs[1:], duration=100, loop=0)

def gif_scale(gif_path, scale:float=2.5):
  gif = Image.open(gif_path)
  resize_frames= [frame.resize((int(frame.width * scale), int(frame.height * scale))) for frame in ImageSequence.Iterator(gif)]
  resize_frames[0].save(gif_path, save_all=True, append_images=resize_frames[1:])
