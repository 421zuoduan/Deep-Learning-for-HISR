
# 基于PSRT和BDT的窗口自适应改进

## 实验代码说明

串行结构


Swin_pool_baseline: 0.418M
Swin_poolv1: head 8, win_size 8, 窗口大小不变, 使用selfattention
Swin_poolv2: head 8, win_size 8, 窗口大小不变, 使用wink_reweight, 0.914M
Swin_poolv3: head 8, win_size 8, 窗口大小不变, 使用wink_reweight, 修改了v2的bug(nW固定)
Swin_poolv4: head 8, win_size 8, 尝试窗口大小与SA相同, 想要nW完全与SA相同, 使用window_reverse
Swin_poolv5: head 8, win_size 8, 窗口大小不变, v3基础上, 只在第二层加入
Swin_poolv6: head 8, win_size 8, 窗口大小不变, v3基础上, 再加入一个池化的全局卷积核



|模型|head|win_size|窗口变小?|epoch|SAM|ERGAS|PSNR|参数量|训练位置|时间|
|----|----|----|----|----|----|----|----|----|----|
|Swin_pool_baseline|8|8|×|1000|2.1187563|1.1237764|51.2758664|0.418M|2号机 UDLv2|20240220|
|Swin_pool_baseline|8|8|×|2000|2.0575124|1.0990003|51.5894557|0.418M|2号机 UDLv2|20240220|
|Swin_poolv3|8|8|×|1000|2.1452999|1.1604029|51.0785415|0.750M|2号机 UDL|20240220|
|Swin_poolv3|8|8|×|1000|2.1313718|1.1403565|51.2451699|0.750M|2号机 UDL|20240220|
|Swin_poolv5|8|8|×|||||0.710M|||
|Swin_poolv6|8|8|×|||||0.768M|||


## 训练说明（笔记本4060）



## 测试结果





## TODO

* 