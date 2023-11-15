
# 基于PSRT和BDT的窗口自适应改进

## 实验代码说明

### 基于BDT的改进

* BDT_KAv1：KernelAttention训练时保持窗口大小不变，但是测试时存在问题；更改代码使得测试时的窗口大小会发生改变

双分支结构：

* BDT_kernelattentionv2：KernelAttention训练时保持窗口数量不变，无法利用金字塔结构的感受野变化的信息
* [code error] BDT_kernelattentionv4：基于v2，KernelAttention去掉了结尾的shortcut和norm。partition维度出错
* [code error] BDT_kernelattentionv6：基于v4，v4中有partition处代码写错了，reverse没改，remake吧:(。重写还是有错
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
* **PSRT_noshuffle**：把PSRT的shuffle都变成普通的Swin Block
* **PSRT_KAv1_noshuffle**：卷积核由池化生成，自注意力、SE计算后去卷全图，卷积核暴力升维（c->c\*\*2），与原图重新计算卷积，1*1卷积融合得到的四张图。并行
* [code error] PSRT_KAv2_noshuffle：把卷窗口的KA放进noshuffle的PSRT中，并行。KAv2的代码有错误，一个维度转换有问题；需要注意，没有for会比有for少三个SE，SE的参数是共享的
* PSRT_KAv3_noshuffle：卷窗口的KA，串行
* PSRT_KAv4_noshuffle：卷窗口的KA，窗口生成卷积核融合成一个全局卷积核（记为global kernel，1\*1卷积实现），窗口卷积核与窗口卷积，全局卷积核与全图卷积，融合得到的五张图为一张图（1*1卷积）。串行
* **PSRT_KAv5_noshuffle**：卷窗口的KA，kernels融合成global kernel，只用global kernel与全图卷积。串行
* PSRT_KAv6_noshuffle：卷窗口的KA，kernels经过自注意力和se后融合成global kernel，窗口核和全局核都卷全局，然后fusion。串行
* **PSRT_KAv7_noshuffle**：基于KAv2和KAv6，尝试解决了维度转换的错误，SE的参数依然是多卷积核共享，无global kernel。窗口生成的卷积核与全图计算卷积，然后融合
* PSRT_KAv8_noshuffle：与KAv6思想相同，se的参数是窗口核和全局核共享
* PSRT_KAv9_noshuffle：基于KAv1，生成卷积核增加c的维度的方法改为repeat，c**2->c\*c后进行一个参数为c*c的linear
* [code error] PSRT_KAv10_noshuffle：基于KAv7，卷全图，不加SE模块，无global kernel。并行
* PSRT_KAv10_noshuffle：基于KAv7，卷全图，有SA无SE，无global kernel。并行
* PSRT_KAv11_noshuffle：基于KAv5和KAv7，卷全图，没有SA和SE，有global kernel。并行
* PSRT_KAv12_noshuffle：基于KAv10和KAv11，卷全图，有SA无SE，有global kernel。并行
* PSRT_KAv13_noshuffle：基于KAv11，卷全图，不加SA，无global kernel。并行，加GELU
* PSRT_KAv14_noshuffle：基于KAv11，SE的激活函数改为GELU；SE放在SA前面；都和reverse后的feature map进行第二次卷积；global kernel由未进行注意力计算的卷积核生成
* PSRT_KAv15_noshuffle：基于KAv14；无SA和SE；都和conv1进行第二次卷积；有global kernel；有shortcut
* PSRT_KAv16_noshuffle：基于KAv5和KAv7，SE的激活函数改为GELU；没有SA有SE，有global kernel。并行
* PSRT_KAv17_noshuffle：基于KAv7和KAv15；无SA和SE；都和原图进行第二次卷积；有global kernel；窗口卷积核使用第一次卷积的window赋权
* PSRT_KAv18_noshuffle：基于KAv7和KAv15；无SA和SE；都和原图进行第二次卷积；无global kernel；窗口卷积核使用第一次卷积的window赋权
* PSRT_KAv19_noshuffle：基于KAv11，卷全图，不加SA和SE，无global kernel。
* PSRT_KAv20_noshuffle：基于KAv17；无SA和SE；都和原图进行第二次卷积；有global kernel；窗口卷积核使用第一次卷积的window赋权；直接用Adaptive pooling


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
|PSRT(embed_Dim=48)|2.2407495|2.4452974|50.0313946|0.538 M|6号机 UDL|20231011|
|PSRT_kernelattentionv5|2.2799347|3.8122486|49.5119861|0.665 M|2号机 UDL|20231015|
|PSRT_KAv1(embed_Dim=48)|2.2844245|2.5096108|49.8647584|0.665 M|2号机 UDL|20231012|
|PSRT_noshuffle(bs=32)|2.1245276|2.2309420|50.4692293|0.538 M|6号机 UDLv2|20231013 断过|
|PSRT_noshuffle(bs=32)|2.1187720|2.1811231|50.4297113|0.538 M|2号机 UDLv2|20231110 没断过|
|PSRT_noshuffle(bs=24)|2.1135997|2.4447264|50.1396261|0.538 M|笔记本|慢慢跑|
|PSRT_KAv1_noshuffle|2.2294778|1.3029419|50.7237681|0.779 M|6号机 UDL|20231017|
|PSRT_KAv2_noshuffle|2.2752936|2.0677896|49.6950313|0.854 M|6号机|20231013|
|PSRT_KAv3_noshuffle|2.2756061|1.7408064|50.1445174|0.918 M|2号机 UDLv2|20231015|
|PSRT_KAv4_noshuffle|2.1899021|2.3440072|50.2209833|1.002 M|2号机 UDLv2|20231018|
|PSRT_KAv5_noshuffle|2.1078129|2.2032974|50.5076604|1.002 M|2号机 UDL|20231019|
|PSRT_KAv6_noshuffle|4.7182505|3.9199647|40.0239899|1.054 M|2号机 UDL|20231022 怀疑过拟合了，2000epoch时，PSNR只有40；1999epoch时，PSNR有50.26；1998epoch时，PSNR有50.43；1500epoch时，PSNR有50.24|
 eta: 0:00:00  SAM: 3.8978894 (avg:4.7182505)  ERGAS: 3.2478695 (avg:3.9199647)  PSNR: 37.3584900 (avg:40.0239899)  again
|PSRT_KAv7_noshuffle|2.1232879|2.1154806|50.4642246|0.894 M|6号机 UDLv2(6太慢了) -> 2号机 UDLv3|20231022|
|PSRT_KAv8_noshuffle|2.1751094|2.4212308|50.3579216|0.946 M|2号机 UDLv2|20231022|
|PSRT_KAv9_noshuffle|2.2132997|3.2366958|50.0673282|0.519 M|6号机 UDL|20231023|
|PSRT_KAv10_noshuffle|2.1785368|1.4475574|50.8828777|0.894 M|2号机 UDL error |20231024|
|PSRT_KAv10_noshuffle|2.2156852|1.4317201|50.7399171|0.894 M|2号机 UDLv2 again|20231103|
|PSRT_KAv11_noshuffle|2.1693590|1.4011621|50.8749442|0.881 M|2号机 UDLv2|20231024|
|PSRT_KAv12_noshuffle|2.3742382|1.2469189|50.6505637|0.851 M|2号机 UDL|20231103|
|PSRT_KAv13_noshuffle|2.1941420|2.4338021|50.1611231|0.894 M|6号机 UDLv2|20231028|
|PSRT_KAv14_noshuffle||||0.851 M||不收敛|
|PSRT_KAv15_noshuffle||||0.890 M|2号机 UDLv3|20231107被kill 20231111不收敛|
|PSRT_KAv16_noshuffle|2.3273963|1.2449526|50.4512170|0.901 M|2号机 UDLv3|20231103 / 20231105|
|PSRT_KAv17_noshuffle|2.2564485|1.4551722|50.7045628|0.884 M|2号机 UDL|20231108 code error|
|PSRT_KAv17_noshuffle||||0.884 M|2号机 UDL|20231111|
|PSRT_KAv18_noshuffle|2.3828535|1.3595995|50.2718298|0.832 M|2号机 UDLv2|20231111|
|PSRT_KAv19_noshuffle|2.5441515|1.3270533|49.6788777|0.832 M|6号机 UDL|20231114|
|PSRT_KAv20_noshuffle||||0.832 M|6号机 UDL|20231115|


## TODO

* v5要把SE去掉重新实验