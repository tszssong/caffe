## 用来训练分类的切图脚本
#### gtTotal: 所有手势的ground truth  
  保存在ssd/gestureDatabyName路径下
  其中five比较特殊，该路径下只保存了vgg_momo，与其他几批汇总如下：  
    1. ssd/gestureDatabyName/5-five-VggMomo-img： 是所有在陌陌录的，vgg标注挑选非紧致的five，包括20181025、20181026粉红色那一批  
        对应label: gtTotal/5-five-VggMomo.txt  
    2. ali1: 宋洋标的非常不紧致的框  
        对应label: gtTotal/5-five_ali1sy.txt  
    3. ali2： 阿里众包第二批紧致框，用于控雨，有五指张开和grab，手势非常乱，整理过几次可见于：  
        https://bitbucket.org/tszs_song/list4gesture/工程目录下的 ‘数据清理过程/ali2five-用于训分类'    
          ali2five_gz_goodGrab.txt             -- 比较像five的控雨动作  
          ali2five_jp-good.txt                       -- 五指张开     
        另外这一批里还有广哲单独挑出来的grab数据，保存在  
        ssd/handgesture5_48G/Tight_ali2five_grab_train-img, 和上述清理过的不重复  
        对应label: gtTotal/T_5_ali2five_6474good  
                         gtTotal/T_5_ali2five_3177grabl  
                         T_5_ali2five_3177grabl/T_5_ali2grab_gz  
    4. ssd/gestureTight4Reg/Tight5-notali2-img： 非阿里众包的紧致框数据, 包括20181025、20181026粉红色那一批  
        对应label: gtTotal/T-5-five_notali2.txt  
      
    

上述数据在网盘中均有备份
