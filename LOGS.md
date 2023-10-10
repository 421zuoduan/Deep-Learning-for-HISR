
# 基于BDT的窗口自适应改进
## 实验效果

* BDT 参数2.656M

* BDT_KAv1 参数3.040M

* BDT_KAv2 参数3.545 M

* BDT_KAv3 参数4.637 M

* BDT_KAv4 参数3.545 M

* BDT_KAv5 参数4.637 M

* BDT_KAv6 参数3.545 M


## 实验代码说明

* BDT_KAv1：KernelAttention训练时保持窗口大小不变，但是测试时存在问题；更改代码使得测试时的窗口大小会发生改变

* BDT_KAv2：KernelAttention训练时保持窗口数量不变，无法利用金字塔结构的感受野变化的信息

* BDT_KAv3：KernelAttention放在Stage的最后，

* [code wrong] BDT_KAv4：KernelAttention去掉了结尾的shortcut和norm，其他与v2保持一致。partition维度出错

* BDT_KAv5：KernelAttention去掉了shortcut（原先有两个shortcut），没有保留layernorm，其他与v3保持一致

* [code wrong] BDT_KAv6：v4中有partition处代码写错了，reverse没改，remake吧:(。重写还是有错

* BDT_KAv7：v4中有reverse改了，partition没改，哥们你真不细心啊。改了重新跑






## 训练说明（笔记本4060）

* 20231009：跑BDT_KAv4，凌晨。运行768epoch，后续放在6号机上运行



## 测试结果

|模型|SAM|ERGAS|PSNR|训练位置|时间|
|----|----|----|----|----|----|
|BDT|2.0513608|1.0617744|51.7136470|2号机|20231008|
|BDT_KAv3|2.0838372|1.0551306|51.6975372|6号机|20231008|
|BDT_KAv4|2.1197825|1.1010958|51.3766497|6号机|20231009|
|BDT_KAv5|2.0677423|1.0793397|51.5658632|2号机|20231009|
|BDT_KAv6|-|-|-|-|-|