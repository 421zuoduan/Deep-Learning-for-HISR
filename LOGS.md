
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
* **PSRT_KAv11_noshuffle**：基于KAv5和KAv7，卷全图，没有SA和SE，有global kernel。并行
* PSRT_KAv12_noshuffle：基于KAv10和KAv11，卷全图，有SA无SE，有global kernel。并行
* PSRT_KAv13_noshuffle：基于KAv11，卷全图，不加SA，无global kernel。并行，加GELU
* PSRT_KAv14_noshuffle：基于KAv11，SE的激活函数改为GELU；SE放在SA前面；都和reverse后的feature map进行第二次卷积；global kernel由未进行注意力计算的卷积核生成
* PSRT_KAv15_noshuffle：基于KAv14；无SA和SE；都和conv1进行第二次卷积；有global kernel；有shortcut
* PSRT_KAv16_noshuffle：基于KAv5和KAv7，SE的激活函数改为GELU；没有SA有SE，有global kernel。并行
* **PSRT_KAv17_noshuffle**：基于KAv7和KAv15；无SA和SE；都和原图进行第二次卷积；有global kernel；窗口卷积核使用第一次卷积的window赋权
* PSRT_KAv18_noshuffle：基于KAv7和KAv15；无SA和SE；都和原图进行第二次卷积；无global kernel；窗口卷积核使用第一次卷积的window赋权
* PSRT_KAv19_noshuffle：基于KAv11，卷全图，不加SA和SE，无global kernel。
* PSRT_KAv20_noshuffle：基于KAv17；无SA和SE；都和原图进行第二次卷积；有global kernel；窗口卷积核使用第一次卷积的window赋权；直接用Adaptive pooling
* PSRT_KAv21_noshuffle：基于KAv17；无SA和SE；都和原图进行第二次卷积；有global kernel；窗口卷积核使用第一次卷积的window赋权；使用7\*7的大核卷积


* PSRT_kernelattentionv5：使用KA的旧代码进行改进（有for）
* PSRT_KAv1：使用重构后的KA代码进行改进（没有for）




### SWAT的模型

embed_dim = 32，bs = 32


**Uformer架构:**

SWAT_baseline：Uformer架构，depth[2, 2, 2, 2, 2]

SWAT_baseline_noshift：Uformer架构，depth[2, 2, 2, 2, 2]

SWAT_baseline_noshiftv2：Restormer架构，depth[2, 2, 2, 2, 2]

SWAT_baseline_noshiftv3：Uformer架构，depth[2, 2, 4, 2, 2]

SWATv1：基于baseline_noshift加KA

SWATv2：基于baseline_noshiftv2加KA

SWATv3：基于baseline_noshiftv3加KA


**SwinIR架构:**

SWAT_baselinev2：SwinIR架构，depth[2, 2, 2, 2, 2]

SWAT_baseline_noshiftv4：基于SWAT_baselinev2, 去掉了shift

SWAT_baseline_noshiftv5：模型最后去掉conv和shortcut

SWATv4: 基于SWAT_baseline_noshiftv4, 加上了kernelattention




**邓尚琦师兄的Swin:**

Swin_baseline: dim=32, depth[2, 4], head[8, 8]

Swinv1

Swin_baselinev2: dim=32, depth[8, 16], head[8, 8]

Swin_baselinev3: dim=48, depth[2, 4], head[8, 8], win_size=8

Swinv3: 窗口逐渐变小, head[8, 8], win_size=4

Swin_baselinev4: dim=48, depth[2, 4], head[16, 16], win_size=8, 窗口在每层逐渐变小

Swinv4: 对v3调参, head[16, 16], win_size=8, 窗口没有变小

Swin_baselinev5: dim=48, depth[2, 4], head[16, 16], win_size=4, 窗口在每层逐渐变小

Swinv5: padding, 分成16窗, 窗口在每层逐渐变小

Swin_baselinev6: dim=48, depth[2, 4], head[16, 16], win_size=16, 窗口在每层逐渐变小

Swinv6: 只在第二层加入

Swinv7: padding, 分成64窗

Swinv8: 基于v6, window_size改为16

Swinv9: 没改好



|模型|SAM|ERGAS|PSNR|参数量|训练位置|时间|
|----|----|----|----|----|----|----|
|SWAT_baseline|1.9349307|0.9883768|52.2688994|1.308 M|2号机 UDLv2|20231124 [1, 2, 4, 4, 2]|
|SWAT_baselinev2|2.0023886|1.0448825|51.7628947|0.613 M|2号机 UDL|20240103|
|SWAT_baseline_noshift|1.9214245|0.9979192|52.1993457|1.269 M|6号机 UDL 改到 2号机 UDL|20231124 [1, 2, 4, 4, 2]|
|SWAT_baseline_noshift|1.9488015|0.9987131|52.1434194|1.269 M|6号机 UDL|20231127 [4, 8, 16, 16, 8]|
|SWATv1|2.6140107|1.5705872|49.5414508|1.964 M|6号机 UDLv2 改到 2号机 UDL nomachine|20231124 1头减少|
|SWATv1|2.0587468|1.1329675|51.4186786|1.964 M|6号机 UDLv2 改到 2号机 UDL nomachine|20231124 4头增加|
|SWATv1|2.0702945|1.4042612|50.8808126|1.964 M|2号机 UDL|20231126 8头不变|
|SWATv1|2.0536631|1.4684086|50.8439539|1.964 M|2号机 UDL|20231126 8头增加|
|SWAT_baseline_noshiftv2|1.9624378|1.0156476|52.0306768|1.072 M|6号机 UDLv2|20231126 [4, 8, 16, 16, 8]|
|SWATv2|2.1123183|1.4603989|50.5582938|1.633 M|2号机 UDLv3|20231126 [4, 8, 16, 16, 8]|
|SWAT_baseline_noshiftv3|2.0066910|1.0203953|51.9469310|1.666 M|6号机 UDLv2|20231129 [4, 8, 16, 16, 8]|
|SWATv3|2.0416935|1.2540732|51.0746963|2.631 M|6号机 UDL|20231129 [4, 8, 16, 16, 8]|
|SWAT_baseline_noshiftv4|2.0467095|1.0404192|51.8539654|0.613 M|6号机 UDLv2|20240103|
|SWATv4|2.1890895|1.1503608|51.1216535|0.906 M|2号机 UDLv2|20240103 head=[8, 8, 8, 8]|
|SWAT_baseline_noshiftv5|2.1827140|1.0912402|51.4073687|0.593 M|2号机 UDLv2|20240104|




|模型|SAM|ERGAS|PSNR|参数量|训练位置|时间|head|win_size|窗口变小?|
|----|----|----|----|----|----|----|----|----|----|
|Swin_baseline|2.0002270|1.0747630|51.8366144|0.910 M|6号机 UDL|20240105 dim=48|8|8|×|
|Swinv1|2.1109537|1.2216327|50.9350288|0.910 M|6号机 UDLv2|20240105 dim=48|
|Swin_baseline|||| M|2号机 UDLv2|20240106 dim=32|
|Swinv1|2.2070388|1.1802504|50.7175754| M|6号机 UDLv2|20240106 dim=32|
|Swin_baselinev3|1.9926484|1.0549250|51.9056820|0.910 M|2号机 UDL|20240108 dim=48|16|16|×|
|Swinv3|2.1027605|1.1144668|51.3873475|1.254 M|6号机 UDLv2|20240108 dim=48|8|4|√|
|Swin_baselinev4|1.9972528|1.0580914|51.9209279|0.910 M|2号机 UDL|20240109 dim=48|16|8|√|
|Swinv4|2.0549133|1.1652680|51.3888834|0.254 M|2号机 UDLv2|20240109 dim=48|16|8|×|
|Swin_baselinev5|2.0304114|1.0900102|51.7110329|0.910 M|2号机 UDLv2|20240111 dim=48|16|4|√|
|Swinv5|2.1011510|1.0906333|51.4140316|1.013 M|2号机 UDL|20240111 dim=48|16|8|√|
|Swinv5_8head|2.1005311|1.0977429|51.4193375|1.013 M|2号机 UDL|20240112 dim=48|16|8|√|
|Swin_baselinev6|2.0006385|1.0753417|51.8400543|0.910 M|2号机 UDLv2|20240117 dim=48|16|16|√|
|Swinv6|2.0768732|1.0771445|51.6581553|0.995 M|2号机 UDLv2|20240114 dim=48|
|Swinv7|2.1605771|1.1705414|51.0369349|1.654 M|2号机 UDL|20240114 dim=48|16|8|√|
|Swinv8|2.0597782|1.0807452|51.6116468|0.995 M|2号机 UDL|20240117 dim=48|16|16|√|



<!-- |SWATv4||||0.906 M|2号机 UDLv2|20240104 head=[4, 4, 4, 4]| -->

SWAT_baseline_noshiftv4     1000epoch SAM: 1.9496609 (avg:2.2695953)  ERGAS: 0.9568591 (avg:1.1029562)  PSNR: 47.8122940 (avg:51.1039127)
SWATv4 2000epoch 51.1216    1000epoch SAM: 2.2170603 (avg:2.3404494)  ERGAS: 1.1031498 (avg:1.2163127)  PSNR: 46.9196053 (avg:50.5814847)

SWAT_baselinev2             1000epoch SAM: 1.9032513 (avg:2.1232411)  ERGAS: 0.9447871 (avg:1.0914433)  PSNR: 47.9477692 (avg:51.3160130) 

SWAT_baseline_noshiftv5     1000epoch SAM: 2.1689320 (avg:2.5721015)  ERGAS: 1.0374856 (avg:1.1901029)  PSNR: 47.1076431 (avg:50.4636959) 





embed_dim = 48，bs = 32

|模型|SAM|ERGAS|PSNR|参数量|训练位置|时间|
|----|----|----|----|----|----|----|
|SWAT_baseline||||2.888 M|||
|SWAT_baseline_noshift||||2.833 M|||
|SWATv1||||4.394 M|||





 
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

#### **baseline**

PSRT设置bs=32，lr=1e-4，embed_dim=48

|模型|SAM|ERGAS|PSNR|参数量|训练位置|时间|
|----|----|----|----|----|----|----|
|PSRT(embed_Dim=32)|-|-|-|0.248 M|-|-|
|PSRT(embed_Dim=64)|-|-|-|0.939 M|-|-|
|PSRT(embed_Dim=48) bs=32|2.2407495|2.4452974|50.0313946|0.538 M|6号机 UDL|20231011|
|PSRT(embed_Dim=48) bs=24|2.1577592|2.5400309|50.1831174|0.538 M|6号机 UDL|20231118|
|PSRT(embed_Dim=48) bs=24|2.2098098|2.8072885|50.1482728|0.538 M|2号机 UDL|20231122|
|PSRT_noshuffle(bs=32)|2.1245276|2.2309420|50.4692293|0.538 M|6号机 UDLv2|20231013 断过|
|PSRT_noshuffle(bs=32)|2.1187720|2.1811231|50.4297113|0.538 M|2号机 UDLv2|20231110 没断过|
|PSRT_noshuffle(bs=32)|2.1176886|2.3832454|50.4341790|0.538 M|6号机 UDLv2|20231122|
|PSRT_noshuffle(bs=24)|2.1135997|2.4447264|50.1396261|0.538 M|笔记本|慢慢跑|
|PSRT_noshuffle(bs=24)|2.1032708|2.2265431|50.4701717|0.538 M|6号机 UDLv2|20231118 没断过|

#### **有global kernel**

|模型|SAM|ERGAS|PSNR|参数量|训练位置|时间|
|----|----|----|----|----|----|----|
|PSRT_KAv5_noshuffle|2.1078129|2.2032974|50.5076604|1.002 M|2号机 UDL|20231019|
|PSRT_KAv6_noshuffle|4.7182505|3.9199647|40.0239899|1.054 M|2号机 UDL|20231022 怀疑过拟合了，2000epoch时，PSNR只有40；1999epoch时，PSNR有50.26；1998epoch时，PSNR有50.43；1500epoch时，PSNR有50.24|
|PSRT_KAv11_noshuffle|2.1693590|1.4011621|50.8749442|0.881 M|2号机 UDLv2|20231024|
|PSRT_KAv12_noshuffle|2.3742382|1.2469189|50.6505637|0.851 M|2号机 UDL|20231103|
|PSRT_KAv16_noshuffle|2.3273963|1.2449526|50.4512170|0.901 M|2号机 UDLv3|20231103 / 20231105|
|PSRT_KAv17_noshuffle|2.2564485|1.4551722|50.7045628|0.884 M|2号机 UDL|20231108 code error|
|PSRT_KAv17_noshuffle|2.1851830|1.2657717|51.0142954|0.884 M|2号机 UDL|20231111|
|PSRT_KAv20_noshuffle|2.5885297|1.3168827|49.7070985|0.832 M|2号机 UDL|20231118|
|PSRT_KAv21_noshuffle|2.2524326|1.8789614|50.2661559|0.554 M|6号机 UDL|20231116|

#### **无global kernel**
|模型|SAM|ERGAS|PSNR|参数量|训练位置|时间|
|----|----|----|----|----|----|----|
|PSRT_KAv7_noshuffle|2.1232879|2.1154806|50.4642246|0.894 M|6号机 UDLv2(6太慢了) -> 2号机 UDLv3|20231022|
|PSRT_KAv8_noshuffle|2.1751094|2.4212308|50.3579216|0.946 M|2号机 UDLv2|20231022|
|PSRT_KAv10_noshuffle|2.1785368|1.4475574|50.8828777|0.894 M|2号机 UDL error |20231024|
|PSRT_KAv10_noshuffle|2.2156852|1.4317201|50.7399171|0.894 M|2号机 UDLv2 again|20231103|
|PSRT_KAv13_noshuffle|2.1941420|2.4338021|50.1611231|0.894 M|6号机 UDLv2|20231028|
|PSRT_KAv18_noshuffle|2.3828535|1.3595995|50.2718298|0.832 M|2号机 UDLv2|20231111|
|PSRT_KAv19_noshuffle|2.5441515|1.3270533|49.6788777|0.832 M|6号机 UDL|20231114|


#### **Conv-GELU-Conv结构**
|模型|SAM|ERGAS|PSNR|参数量|训练位置|时间|
|----|----|----|----|----|----|----|
|PSRT_KAv14_noshuffle||||0.851 M||不收敛|
|PSRT_KAv15_noshuffle||||0.890 M|2号机 UDLv3|20231107被kill 20231111不收敛|


#### **卷窗口**
|模型|SAM|ERGAS|PSNR|参数量|训练位置|时间|
|----|----|----|----|----|----|----|
|PSRT_KAv2_noshuffle|2.2752936|2.0677896|49.6950313|0.854 M|6号机|20231013|
|PSRT_KAv3_noshuffle|2.2756061|1.7408064|50.1445174|0.918 M|2号机 UDLv2|20231015|
|PSRT_KAv4_noshuffle|2.1899021|2.3440072|50.2209833|1.002 M|2号机 UDLv2|20231018|


#### **池化生成卷积核**
|模型|SAM|ERGAS|PSNR|参数量|训练位置|时间|
|----|----|----|----|----|----|----|
|PSRT_KAv1_noshuffle|2.2294778|1.3029419|50.7237681|0.779 M|6号机 UDL|20231017|
|PSRT_KAv9_noshuffle|2.2132997|3.2366958|50.0673282|0.519 M|6号机 UDL|20231023|


#### **PSRT的改进**
|模型|SAM|ERGAS|PSNR|参数量|训练位置|时间|
|----|----|----|----|----|----|----|
|PSRT|2.2407495|2.4452974|50.0313946|0.538 M|6号机 UDL|20231011|
|PSRT_KAv11|2.1810421|2.8618149|50.2121601|0.653 M|6号机 UDL nomachine|20231119|
|PSRT_KAv17|2.3413245|2.7532913|49.0937249|0.653 M|6号机 UDL|20231119|
|PSRT_KAv17_allinsert|2.3304226|1.5883428|50.3411182|0.884 M|6号机 UDL|20231119|


我也忘了以下是什么了

|模型|SAM|ERGAS|PSNR|参数量|训练位置|时间|
|----|----|----|----|----|----|----|
|PSRT_kernelattentionv5|2.2799347|3.8122486|49.5119861|0.665 M|2号机 UDL|20231015|
|PSRT_KAv1(embed_Dim=48)|2.2844245|2.5096108|49.8647584|0.665 M|2号机 UDL|20231012|


### PSRT模型改进的测试结果





## TODO

* v5要把SE去掉重新实验
* 遥感数据集一般都是正方形的吗