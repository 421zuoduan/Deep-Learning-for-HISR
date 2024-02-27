
# 基于PSRT和BDT的窗口自适应改进

## 实验代码说明

串行结构


### 基础实验(wink_reweight, global_kernel, 全加还是仅第二层加, shift时加不加)

Swin_pool_baseline: 0.418M

Swin_poolv1: head 8, win_size 8, 窗口大小不变, 使用selfattention

Swin_poolv2: head 8, win_size 8, 窗口大小不变, 使用wink_reweight, 0.914M

Swin_poolv3: head 8, win_size 8, 窗口大小不变, 使用wink_reweight, 修改了v2的bug(nW固定)

Swin_poolv4: head 8, win_size 8, 尝试窗口大小与SA相同, 想要nW完全与SA相同, 使用window_reverse, **没写完**

Swin_poolv5: head 8, win_size 8, 窗口大小不变, v3基础上, 只在第二层加入

Swin_poolv6: head 8, win_size 8, 窗口大小不变, v3基础上, 再加入一个池化的全局卷积核

Swin_poolv7: head 8, win_size 8, 窗口大小不变, 结合v5和v6, 只在第二层加入模块, 再加入一个池化的全局卷积核

**Swin_poolv12**: head 8, win_size 8, 窗口大小不变, v7基础上, 仅在不shift的窗口上操作, 有池化全局卷积核, 只在第二层加入

Swin_poolv16: head 8, win_size 8, 窗口大小不变, v12基础上, 改变模块加入的位置, 在attention前加

Swin_poolv27: head 8, win_size 8, 窗口大小不变, v10基础上, 去掉聚合的global kernel, 仅使用池化的global kernel


### 基于Swin_poolv10的实验

Swin_poolv8: head 8, win_size 8, 窗口大小不变, v3基础上丢弃不重要的窗口再聚合成global kernel, **没写完**

Swin_poolv9: head 8, win_size 8, 窗口大小不变, 与v3对比, 没有使用wink_reweight

**Swin_poolv10**: head 8, win_size 8, 窗口大小不变, v3基础上, 仅在不shift的窗口上操作

Swin_poolv11: head 8, win_size 8, 窗口大小不变, v10基础上, 只在第二层加入

Swin_poolv13: head 8, win_size 8, 窗口大小不变, v10基础上, 修改wink_reweight, 增加卷积核在通道和空间上共享的SE

**Swin_poolv15**: head 8, win_size 8, 窗口大小不变, v13基础上, 修改wink_reweight, 卷积核仅在通道上SE, 窗口池化前进行通道共享的SE

Swin_poolv17: head 8, win_size 8, 窗口大小不变, v13基础上, SE的mlp改为先降维再升维

Swin_poolv14: head 8, win_size 8, 窗口大小不变, v10基础上, 64窗口

Swin_poolv18: head 8, win_size 8, 窗口大小不变, v10基础上, 加入attn_map池化后的自注意力

Swin_poolv19: head 8, win_size 8, 窗口大小不变, v10基础上, fusion改成分组卷积

Swin_poolv21: head 8, win_size 8, 窗口大小不变, v10基础上, fusion时丢弃softmax后权重低的窗口卷积核


### 分组卷积聚合信息

Swin_poolv22: head 8, win_size 8, 窗口大小不变, v10基础上, 使用分组卷积聚合global kernel

Swin_poolv23: head 8, win_size 8, 窗口大小不变, v22基础上, 增加一个池化的全局卷积核

Swin_poolv24: head 8, win_size 8, 窗口大小不变, v23基础上, 结合v15改变win_reweight, 修改代码细节

Swin_poolv24_nopoolgk: head 8, win_size 8, 窗口大小不变, v22基础上, 结合v15改变win_reweight, 修改代码细节, 没有使用池化全局卷积核

Swin_poolv25: head 8, win_size 8, 窗口大小不变, v22groupconvfusion基础上, 变成双分支结构, 每个分支均为C, 使用分组卷积聚合两张feature map

Swin_poolv26: head 8, win_size 8, 窗口大小不变, v25基础上, 将分组卷积聚合改成普通卷积


### 减小窗口大小

Swin_pool_baselinev3: head 8, win_size 8, 同一层窗口大小减少

Swin_poolv20: head 8, win_size 8, 基于v10, 同一层窗口大小减少


### 卷积生成qkv的改进

Swin_pool_baselinev2: windowattention上, 将qkv生成方式改为dwconv

Swin_qkvv1: head 8, win_size 8, 窗口大小不变, baseline基础上, 加入Swinv8的内容

Swin_qkvv2: head 8, win_size 8, 窗口大小不变, baseline基础上, 加入Swin_poolv10的内容, 卷积通过池化生成

Swin_qkvv3: head 8, win_size 8, 窗口大小不变, v3基础上, 计入GELU

Swin_qkvv4: head 8, win_size 8, 窗口大小不变, v2基础上, 直接替换原有分组卷积



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
|Swin_poolv11|8|8|×|2000|2.0973329|1.1246392|51.4355878|0.564M|6号机 UDL|20240222|
|Swin_poolv12|8|8|×|2000|2.0703675|1.1244291|51.4649311|0.572M|6号机 UDLv2|20240222|
|Swin_poolv13|8|8|×|2000|2.0888323|1.1220643|51.4287096|0.644M|6号机 UDL|20240223|
|Swin_poolv14|8|8|×|2000|2.1152405|1.1676570|51.1290873|1.127M|2号机 UDL|20240223|
|Swin_poolv15|8|8|×|2000|2.1842559|1.1034097|51.4900409|0.640M|6号机 UDL|20240223|
|Swin_poolv16|8|8|×|2000|2.0916659|1.1376898|51.3469478|0.572M|2号机 UDL|20240223|
|Swin_poolv27|8|8|×|2000||||0.575M|2号机 UDLv3|20240227|
|Swin_poolv17|8|8|×|2000|2.0884990|1.1169446|51.3982006|0.586M|6号机 UDLv2|20240223|
|Swin_poolv18|8|8|×|2000|2.1567486|1.1810000|51.0778535|2.080M|6号机 UDL|20240224|
|Swin_poolv19|8|8|×|2000|2.0977861|1.1031732|51.4121708|0.430M|6号机 UDLv2|20240224|
|Swin_poolv21|8|8|×|2000|2.1703069|1.1648251|50.9806678|0.584M|2号机 UDLv2|20240224|



|Swin_poolv22_normalconv|8|8|×|2000|2.1610719|1.1718743|51.1293605|0.732M|6号机 UDL|20240227|
|Swin_poolv22_groupconv|8|8|×|2000|2.1052242|1.1123530|51.4093260|0.587M|2号机 UDL|20240227|
|Swin_poolv22_groupconvfusion|8|8|×|2000|2.0980517|1.0974184|51.5346839|0.433M|6号机 UDL|20240227|
|Swin_poolv23|8|8|×|2000|2.0731021|1.1599144|51.3773082|0.433M|6号机 UDL|20240227|
|Swin_poolv24|8|8|×|2000||||0.508M|2号机 UDL|20240227|
|Swin_poolv24|8|8|×|2000||||0.507M|6号机 UDL|20240227|
|Swin_poolv25|8|8|×|2000||||0.433M|2号机 UDLv2|20240227|
|Swin_poolv25|8|8|×|2000||||0.451M|2号机 UDLv2|20240227|


|Swin_pool_baselinev2|8|8|×|2000|2.1143289|1.1144895|51.3939195|0.428M|2号机 UDL|20240224|
|Swin_qkvv2|8|8|×|2000|2.2355720|1.2335357|50.6819552|0.926M|2号机 UDL|20240224|
|Swin_qkvv3|8|8|×|2000|2.2309780|1.2588267|50.6825704|0.926M|2号机 UDL|20240224|
|Swin_qkvv4|8|8|×|2000|2.2133673|1.2244732|50.7088162|0.921M|6号机 UDLv2|20240226|
|Swin_pool_baselinev3|8|8|×|2000|2.0955985|1.0871105|51.4333194|0.418M|2号机 UDL|20240226|
|Swin_poolv20|8|8|×|2000|2.1272049|1.1356858|51.2372097|0.657M|6号机 UDL|20240226|

## 万一有用呢

1. QKV的循环
2. QKV用到模块里
3. 不同的头融入不同的块, 类似MoE
4. softmax后, 小的就不要了, 然后fusion


## TODO

* 多看论文
* 看PSRT的代码还有什么不同