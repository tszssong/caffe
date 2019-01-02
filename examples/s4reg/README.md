## 回归切图脚本  
#### gt开头的路径里是切图对应list  
gt：一些零散的list  
gt_sep: 13类线上手势list,其中绿布单独放在一个路径了  
gt_test: 王司4666张图片测试集+晓鹏ali2控雨20段视频4511张  
gt_13cls_withGreen：gt_sep的绿布手势合并版本  

#### 脚本：  
travDir.sh： 后台执行脚本工具  
crop4reg.py:   caffe离线训练切图，给定需要的小图数量，通过翻转、高斯模糊等方式增广到要求的数量停止  
crop4test.py:  caffe离线训练切图,用于切测试图片
  
crop300.py:  pytorch切图脚本， 每张图只切一次，如果切出来的小图有其他手，将小图里的其他手涂黑   
draw300.py: 在小图上画框，用来验证切图label是否正确。  
                     由于后来在pytorch工程中加入画label程序，这个脚本基本不用了  
                       
writePickel.py: 用于解决caffe离线小图读入速度太慢问题
