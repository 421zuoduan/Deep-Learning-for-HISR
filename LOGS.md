
# 基于BDT的窗口自适应改进
## 实验效果

* BDT 参数2.656M

* BDT_KAv1 参数3.040M

* BDT_KAv2 参数3.545 M


## 实验代码说明

* BDT_KAv1：KernelAttention训练时保持窗口大小不变，但是测试时存在问题；更改代码使得测试时的窗口大小会发生改变

* BDT_KAv2：KernelAttention训练时保持窗口数量不变，无法利用金字塔结构的感受野变化的信息

* BDT_KAv3：KernelAttetion放在Stage的最后

* BDT_KAv4：KernelAttetion去掉了结尾的shortcut和norm，其他与v2保持一致

* BDT_KAv5：KernelAttetion，其他与v3保持一致






## 训练说明（笔记本4060）

* 20231009：跑BDT_KAv4，凌晨。运行768epoch，后续放在6号机上运行



## 测试结果

|模型|SAM|ERGAS|PSNR|训练位置|时间|
|----|----|----|----|----|----|
|BDT|2.0513608|1.0617744|51.7136470|2号机|20231008|
|BDT_KAv3|2.0838372|1.0551306|51.6975372|6号机|20231008|
|BDT_KAv4|||||