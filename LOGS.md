
# 基于BDT的窗口自适应改进
## 实验效果

* BDT 参数2.656M

* BDT_KAv1 参数3.040M

* BDT_KAv2 参数3.545 M


## 实验代码说明

* BDT_KAv1：KernelAttention训练时保持窗口大小不变，但是测试时存在问题；更改代码使得测试时的窗口大小会发生改变

* BDT_KAv2：KernelAttention训练时保持窗口数量不变，无法利用金字塔结构的感受野变化的信息

* BDT_KAv3：KernelAttetion放在Stage的最后