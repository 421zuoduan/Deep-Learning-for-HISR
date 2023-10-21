
# 基于PSRT和BDT的窗口自适应改进

## 实验代码说明

### 基于BDT的改进

* BDT_KAv1：KernelAttention训练时保持窗口大小不变，但是测试时存在问题；更改代码使得测试时的窗口大小会发生改变

双分支结构：

* BDT_kernelattentionv2：KernelAttention训练时保持窗口数量不变，无法利用金字塔结构的感受野变化的信息
* [code wrong] BDT_kernelattentionv4：基于v2，KernelAttention去掉了结尾的shortcut和norm。partition维度出错
* [code wrong] BDT_kernelattentionv6：基于v4，v4中有partition处代码写错了，reverse没改，remake吧:(。重写还是有错
* BDT_kernelattentionv7：基于v4，v4中有reverse改了，partition没改，哥们你真不细心啊。改了重新跑
* BDT_kernelattentionv8：基于v7，改变KernelAttention里的窗口数均为16

module放在stage最后:

* BDT_kernelattentionv3：KernelAttention放在Stage的最后
* BDT_kernelattentionv5：基于v3，KernelAttention去掉了shortcut（原先有两个shortcut），没有保留layernorm
* BDT_kernelattentionv9：基于v5，KernelAttention前要加一个norm

在双分支上重新写代码：

* BDT_KAv1：试图实现对整张图卷积，还没有写完
* BDT_KAv2：KA代码重构完成，后续的KA可以从这里调取





### 基于PSRT的改进

目前先做双分支吧
* PSRT_noshuffle：把shuffle都变成普通的Swin Block
* PSRT_KAv1_noshuffle：卷积核由池化生成，自注意力计算后去卷全图，，但是是新写法
* PSRT_KAv2_noshuffle：把卷局部的KA放进noshuffle的PSRT中，没有for
* PSRT_KAv3_noshuffle：卷局部的KA，有for
* PSRT_KAv4_noshuffle：卷局部、卷全局进行fusion，有for
* PSRT_KAv5_noshuffle：局部生成kernel，kernel聚合成global kernel，只用global kernel与全图卷积
* PSRT_KAv6_noshuffle：去掉了Norm，（代码仍需改进）
* 
* PSRT_kernelattentionv5：使用KA的旧代码进行改进（有for）
* PSRT_KAv1：使用重构后的KA代码进行改进（没有for）





## 训练说明（笔记本4060）

* 20231009：跑BDT_KAv4，凌晨。运行768epoch，后续放在6号机上运行





## 测试结果

### BDT模型改进的测试结果

BDT设bs=64，lr=2e-4。后续需要重新实验，设置bs=32，lr=1e-4

|模型|SAM|ERGAS|PSNR|参数量|训练位置|时间|
|----|----|----|----|----|----|----|
|BDT|2.0513608|1.0617744|51.7136470|2.656M|2号机|20231008|
|BDT_kernelattentionv3|2.0838372|1.0551306|51.6975372|4.637 M|6号机|20231008|
|BDT_kernelattentionv4|2.1197825|1.1010958|51.3766497|3.545 M|6号机|20231009|
|BDT_kernelattentionv5|2.0677423|1.0793397|51.5658632|4.637 M|2号机|20231009|
|BDT_kernelattentionv6|-|-|-|-|-|
|BDT_kernelattentionv7|2.0654192|1.0971936|51.4890792|3.545 M|6号机|20231010|
|BDT_kernelattentionv8|-|-|-|-|-|
|BDT_kernelattentionv9|2.3876350|1.1874575|50.5793433|4.637 M|2号机|20231010|
|BDT_KAv1||||4.617 M|||
|BDT_KAv2||||3.350 M|||


### PSRT模型改进的测试结果

PSRT设置bs=32，lr=1e-4，embed_dim=48

|模型|SAM|ERGAS|PSNR|参数量|训练位置|时间|
|----|----|----|----|----|----|----|
|PSRT(embed_Dim=32)|-|-|-|0.248 M|-|-|
|PSRT(embed_Dim=64)|-|-|-|0.939 M|-|-|
|PSRT(embed_Dim=48)|2.2407495|2.4452974|50.0313946|0.538 M|6号机|20231011|
|PSRT_kernelattentionv5|2.2799347|3.8122486|49.5119861|0.665 M|2号机 UDL|20231015|
|PSRT_KAv1(embed_Dim=48)|2.2844245|2.5096108|49.8647584|0.665 M|2号机 UDL|20231012|
|PSRT_noshuffle|2.1245276|2.2309420|50.4692293|0.538 M|6号机|20231013|
|PSRT_KAv1_noshuffle|2.2294778|1.3029419|50.7237681|0.779 M|6号机 UDL|20231017|
|PSRT_KAv2_noshuffle|2.2752936|2.0677896|49.6950313|0.854 M|6号机|20231013|
|PSRT_KAv3_noshuffle|2.2756061|1.7408064|50.1445174|0.918 M|2号机 UDLv2|20231015|
|PSRT_KAv4_noshuffle|2.1899021|2.3440072|50.2209833|1.002 M|2号机 UDLv2|20231018|
|PSRT_KAv5_noshuffle|2.1078129|2.2032974|50.5076604|1.002 M|2号机 UDL|20231019|