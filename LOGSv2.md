
# 基于PSRT和BDT的窗口自适应改进

## 实验代码说明

串行结构


Swin_pool_baseline: 0.418M

Swin_poolv1: head 8, win_size 8, 窗口大小不变, 使用selfattention

Swin_poolv2: head 8, win_size 8, 窗口大小不变, 使用wink_reweight, 0.914M

Swin_poolv3: head 8, win_size 8, 窗口大小不变, 使用wink_reweight, 修改了v2的bug(nW固定)

Swin_poolv4: head 8, win_size 8, 尝试窗口大小与SA相同, 想要nW完全与SA相同, 使用window_reverse, **没写完**

Swin_poolv5: head 8, win_size 8, 窗口大小不变, v3基础上, 只在第二层加入

Swin_poolv6: head 8, win_size 8, 窗口大小不变, v3基础上, 再加入一个池化的全局卷积核

Swin_poolv7: head 8, win_size 8, 窗口大小不变, 结合v5和v6, 只在第二层加入模块, 再加入一个池化的全局卷积核

Swin_poolv8: head 8, win_size 8, 窗口大小不变, v3基础上丢弃不重要的窗口再聚合成global kernel, **没写完**

Swin_poolv9: head 8, win_size 8, 窗口大小不变, 与v3对比, 没有使用wink_reweight

Swin_poolv10: head 8, win_size 8, 窗口大小不变, v3基础上, 仅在不shift的窗口上操作

Swin_poolv11: head 8, win_size 8, 窗口大小不变, v10基础上, 只在第二层加入

Swin_poolv12: head 8, win_size 8, 窗口大小不变, v7基础上, 仅在不shift的窗口上操作


|模型|head|win_size|窗口变小?|epoch|SAM|ERGAS|PSNR|参数量|训练位置|时间|
|----|----|----|----|----|----|----|----|----|----|----|
|Swin_pool_baseline|8|8|×|1000|2.1187563|1.1237764|51.2758664|0.418M|2号机 UDLv2|20240220|
|Swin_pool_baseline|8|8|×|2000|2.0575124|1.0990003|51.5894557|0.418M|2号机 UDLv2|20240220|
|Swin_poolv3|8|8|×|1000|2.1452999|1.1604029|51.0785415|0.750M|2号机 UDL|20240220|
|Swin_poolv3|8|8|×|1000|2.1313718|1.1403565|51.2451699|0.750M|2号机 UDL|20240220|
|Swin_poolv5|8|8|×|2000|2.1330074|1.1528373|51.2376189|0.710M||0.710M|2号机 UDL|20240220|
|Swin_poolv6|8|8|×|2000|2.0602730|1.0984690|51.5743276|0.768M|2号机 UDLv2|20240220 中间断了, 代码是baseline跑错了|
|Swin_poolv6|8|8|×|2000|2.0575124|1.0990003|51.5894557|0.768M|2号机 UDLv2|20240221 重新跑, 代码是baseline跑错了|
|Swin_poolv6|8|8|×|2000|2.1081462|1.1541255|51.1988231|0.768M|2号机 UDLv2|20240222|
|Swin_poolv7|8|8|×|2000|2.1234807|1.1642470|51.2360521|0.726M|2号机 UDL|20240221|
|Swin_poolv9|8|8|×|2000|2.1238296|1.1451240|51.2528690|0.732M|6号机 UDL|20240221|
|Swin_poolv10|8|8|×|2000|2.0853686|1.1198790|51.4375870|0.584M|6号机 UDL|20240222|
|Swin_poolv11|8|8|×|2000||||0.564M|6号机 UDL|20240222|
|Swin_poolv12|8|8|×|2000||||0.572M|6号机 UDL|20240222|



## 训练说明（笔记本4060）



## 测试结果





## TODO

* 