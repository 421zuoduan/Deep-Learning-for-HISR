import torch.utils.data as data
import torch
import h5py
import numpy as np


class H5Dataset(data.Dataset):
    def __init__(self, file_path):
        super(H5Dataset, self).__init__()
        self.file_path = file_path
        self.dataset = None
        with h5py.File(self.file_path, 'r') as f:
            self.dataset_len = len(f["gt"])
    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')
        #TypeError: can't convert np.ndarray of type numpy.longdouble.
        # The only supported types are: float64, float32, float16, int64, int32, int16, int8, uint8, and bool
        # self.gt = torch.from_numpy(self.dataset.get("gt")[...].astype(np.float32)) / 2047.0  # NxCxHxW
        # self.ms = torch.from_numpy(self.dataset.get("ms")[...].astype(np.float32)) / 2047.0
        # self.lms = torch.from_numpy(self.dataset.get("lms")[...].astype(np.float32)) / 2047.0
        # self.pan = torch.from_numpy(self.dataset.get("pan")[...].astype(np.float32)) / 2047.0
        # return self.gt[index, :, :, :], \
        #        self.lms[index, :, :, :], \
        #        self.ms[index, :, :, :], \
        #        self.pan[index, :, :, :]

    def __len__(self):
        return self.dataset_len


class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        dataset = h5py.File(file_path, 'r')
        try:
            self.gt = torch.from_numpy(dataset["gt"][...]).float() / 2047.0  # NxCxHxW
            self.ms = torch.from_numpy(dataset["ms"][...]).float() / 2047.0
            self.lms = torch.from_numpy(dataset["lms"][...]).float() / 2047.0
            self.pan = torch.from_numpy(dataset["pan"][...]).float() / 2047.0
        except Exception:
            self.gt = np.array(dataset["gt"][...], dtype=np.float32)
            self.ms = np.array(dataset["ms"][...], dtype=np.float32)
            self.lms = np.array(dataset["lms"][...], dtype=np.float32)
            self.pan = np.array(dataset["pan"][...], dtype=np.float32)
            self.gt = torch.from_numpy(self.gt)
            self.ms = torch.from_numpy(self.ms)
            self.lms = torch.from_numpy(self.lms)
            self.pan = torch.from_numpy(self.pan)
            # # gt = dataset["gt"]
            # gt = dataset.get("gt")
            # gt = gt[...]
            # try:
            #     gt_t = torch.from_numpy(gt)
            # except:
            #     print(11111)
            #     gt_t = torch.from_numpy(gt)
            # gt_tf = gt_t.float()

        print("loading data: \n"
              "gt:     {} \n"
              "lms:    {} \n"
              "ms_hp:  {} \n"
              "pan_hp: {} \n".format(self.gt.size(), self.lms.size(), self.ms.size(), self.pan.size()))

    #####必要函数
    def __getitem__(self, index):
        return self.gt[index, :, :, :], \
               self.lms[index, :, :, :], \
               self.ms[index, :, :, :], \
               self.pan[index, :, :, :]

    #####必要函数
    def __len__(self):
        return self.gt.shape[0]



class DatasetFromHdf5_MS(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5_MS, self).__init__()
        dataset = h5py.File(file_path, 'r')
        try:
            self.gt = torch.from_numpy(dataset["gt"][...]).float() / 2047.0  # NxCxHxW
            self.ms = torch.from_numpy(dataset["ms"][...]).float() / 2047.0
            self.mms = torch.nn.functional.interpolate(self.ms, size=(self.ms.size(2)*2, self.ms.size(3)*2), mode="bilinear", align_corners=True)
            self.lms = torch.from_numpy(dataset["lms"][...]).float() / 2047.0
            self.pan = torch.from_numpy(dataset["pan"][...]).float() / 2047.0
        except Exception:
            self.gt = np.array(dataset["gt"][...], dtype=np.float32) / 2047.0
            self.ms = np.array(dataset["ms"][...], dtype=np.float32) / 2047.0
            self.lms = np.array(dataset["lms"][...], dtype=np.float32) / 2047.0
            self.pan = np.array(dataset["pan"][...], dtype=np.float32) / 2047.0
            self.gt = torch.from_numpy(self.gt)
            self.ms = torch.from_numpy(self.ms)
            self.mms = torch.nn.functional.interpolate(self.ms, size=(self.ms.size(2)*2, self.ms.size(3)*2), mode="bilinear", align_corners=True)
            self.lms = torch.from_numpy(self.lms)
            self.pan = torch.from_numpy(self.pan)
            # # gt = dataset["gt"]
            # gt = dataset.get("gt")
            # gt = gt[...]
            # try:
            #     gt_t = torch.from_numpy(gt)
            # except:
            #     print(11111)
            #     gt_t = torch.from_numpy(gt)
            # gt_tf = gt_t.float()

        print("loading data: \n"
              "gt:     {} \n"
              "lms:    {} \n"
              "mms_hp: {} \n"
              "ms_hp:  {} \n"
              "pan_hp: {} \n".format(self.gt.size(), self.lms.size(), self.mms.size(),self.ms.size(), self.pan.size()))

    #####必要函数
    def __getitem__(self, index):
        return self.gt[index, :, :, :], \
               self.lms[index, :, :, :], \
               self.mms[index, :, :, :], \
               self.ms[index, :, :, :], \
               self.pan[index, :, :, :]

    #####必要函数
    def __len__(self):
        return self.gt.shape[0]



import time

def timer(func):
    def performer():
        start = time.time()
        out = func()
        print("time: ", time.time() - start)
        return out
    return performer


@timer
def test():
    # 记录加载时间，并尽快缩短，h5已经提高了gpu利用率，但仍会受限transform操作和cpu单核频率


    start = time.time()
    train_set = DatasetFromHdf5('../training_data/train.h5')  # creat data for training
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    validate_set = DatasetFromHdf5('../training_data/valid.h5')  # creat data for validation
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    print("time: ", time.time() - start)


if __name__ == "__main__":
    # 记录加载时间，并尽快缩短，h5已经提高了gpu利用率，但仍会受限transform操作和cpu单核频率
    from torch.utils.data import DataLoader

    batch_size = 32

    start = time.time()
    train_set = DatasetFromHdf5_MS('../training_data/train.h5')  # creat data for training
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    validate_set = DatasetFromHdf5_MS('../training_data/valid.h5')  # creat data for validation
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    print("time: ", time.time() - start)
    for iteration, batch in enumerate(training_data_loader, 1):

        print(iteration, "time: ", time.time() - start)

    # for i in range(10):
    #     print("iteration: ", i)
    #     # train_set = DatasetFromHdf5_MS('../training_data/train.h5')  # creat data for training
    #     # training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
    #     #                                   pin_memory=True,
    #     #                                   drop_last=True)  # put training data to DataLoader for batches
    #     #
    #     # validate_set = DatasetFromHdf5_MS('../training_data/valid.h5')  # creat data for validation
    #     # validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
    #     #                                   pin_memory=True,
    #     #                                   drop_last=True)  # put training data to DataLoader for batches
    #     # next(iter(training_data_loader)), next(iter(validate_data_loader))
    #     # del train_set, training_data_loader
    #     # del validate_set, validate_data_loader
    #     test()

'''
D:\anaconda\envs\ai\python.exe D:/ProjectSets/Group/02-PanHRNet-Pytorch/data.py
loading data: 
gt:     [Nx8x64x64] 
lms:    [Nx8x64x64] 
ms_hp:  [Nx8x16x16] 
pan_hp: [Nx1x64x64] 

loading data: 
gt:     [Nx8x64x64] 
lms:    [Nx8x64x64] 
ms_hp:  [Nx8x16x16] 
pan_hp: [Nx1x64x64] 

time:  0.0019986629486083984
1 time:  1.3855109214782715
2 time:  1.4215154647827148
3 time:  1.4505438804626465
4 time:  1.4845147132873535
5 time:  1.5135211944580078
6 time:  1.543511152267456
7 time:  1.5745160579681396
8 time:  1.6065120697021484
9 time:  1.635514497756958
10 time:  1.6645145416259766
11 time:  1.6985139846801758
12 time:  1.7295372486114502
13 time:  1.7610363960266113
14 time:  1.7900373935699463
15 time:  1.8210375308990479
16 time:  1.8490395545959473
17 time:  1.8780386447906494
18 time:  1.9080440998077393
19 time:  1.9380385875701904
20 time:  1.96903657913208
21 time:  1.9980366230010986
22 time:  2.0290415287017822
23 time:  2.060037851333618
24 time:  2.0920400619506836
25 time:  2.121037006378174
26 time:  2.1510653495788574
27 time:  2.1820385456085205
28 time:  2.213048219680786
29 time:  2.2420380115509033
30 time:  2.273040533065796
31 time:  2.3050403594970703
32 time:  2.337040901184082
33 time:  2.3690454959869385
34 time:  2.401038885116577
35 time:  2.4320411682128906
36 time:  2.4630565643310547
37 time:  2.4950385093688965
38 time:  2.5280377864837646
39 time:  2.561038017272949
40 time:  2.593061685562134
41 time:  2.62504243850708
42 time:  2.6550793647766113
43 time:  2.690037488937378
44 time:  2.721043586730957
45 time:  2.750042200088501
46 time:  2.7830419540405273
47 time:  2.812058925628662
48 time:  2.846038579940796
49 time:  2.878038167953491
50 time:  2.909043312072754
51 time:  2.938042402267456
52 time:  2.9690358638763428
53 time:  2.999049186706543
54 time:  3.0340864658355713
55 time:  3.068086862564087
56 time:  3.099086046218872
57 time:  3.128098487854004
58 time:  3.1591341495513916
59 time:  3.191088914871216
60 time:  3.2250874042510986
61 time:  3.255087375640869
62 time:  3.290109872817993
63 time:  3.3210854530334473
64 time:  3.3510873317718506
65 time:  3.3800878524780273
66 time:  3.4140889644622803
67 time:  3.4450864791870117
68 time:  3.479093313217163
69 time:  3.511120080947876
70 time:  3.5420870780944824
71 time:  3.5720880031585693
72 time:  3.6060879230499268
73 time:  3.6351165771484375
74 time:  3.669095277786255
75 time:  3.705087423324585
76 time:  3.7400903701782227
77 time:  3.7730906009674072
78 time:  3.8070855140686035
79 time:  3.840087413787842
80 time:  3.869112253189087
81 time:  3.901090621948242
82 time:  3.930093765258789
83 time:  3.9610869884490967
84 time:  3.994112253189087
85 time:  4.024094104766846
86 time:  4.05508828163147
87 time:  4.089087247848511
88 time:  4.12108850479126
89 time:  4.152093172073364
90 time:  4.182088375091553
91 time:  4.216091871261597
92 time:  4.249087333679199
93 time:  4.280087471008301
94 time:  4.313090085983276
95 time:  4.344084739685059
96 time:  4.375086784362793
97 time:  4.408093214035034
98 time:  4.439088582992554
99 time:  4.470089435577393
100 time:  4.50308632850647
101 time:  4.535092830657959
102 time:  4.56711483001709
103 time:  4.603086233139038
104 time:  4.635086297988892
105 time:  4.6680896282196045
106 time:  4.700091600418091
107 time:  4.730112791061401
108 time:  4.762089490890503
109 time:  4.792086362838745
110 time:  4.821086406707764
111 time:  4.855086326599121
112 time:  4.8840906620025635
113 time:  4.915086269378662
114 time:  4.945085763931274
115 time:  4.975086688995361
116 time:  5.005084753036499
117 time:  5.034085512161255
118 time:  5.06560492515564
119 time:  5.095635175704956
120 time:  5.127607822418213
121 time:  5.158600568771362
122 time:  5.187611818313599
123 time:  5.217602014541626
124 time:  5.248650074005127
125 time:  5.278603792190552
126 time:  5.3106019496917725
127 time:  5.340601682662964
128 time:  5.3766114711761475
129 time:  5.405616044998169
130 time:  5.437606334686279
131 time:  5.466606378555298
132 time:  5.499606370925903
133 time:  5.528605699539185
134 time:  5.560600996017456
135 time:  5.591602087020874
136 time:  5.6246020793914795
137 time:  5.6556031703948975
138 time:  5.685606956481934
139 time:  5.714602470397949
140 time:  5.746600389480591
141 time:  5.778619289398193
142 time:  5.809601306915283
143 time:  5.842118978500366
144 time:  5.871150255203247
145 time:  5.902124881744385
146 time:  5.932116746902466
147 time:  5.963123798370361
148 time:  5.997135162353516
149 time:  6.0261313915252686
150 time:  6.058121681213379
151 time:  6.088123559951782
152 time:  6.118119239807129
153 time:  6.147122859954834
154 time:  6.1811203956604
155 time:  6.21612024307251
156 time:  6.248116731643677
157 time:  6.282153606414795
158 time:  6.315641641616821
159 time:  6.347653865814209
160 time:  6.3776373863220215
161 time:  6.409637928009033
162 time:  6.442633628845215
163 time:  6.475642442703247
164 time:  6.5066328048706055
165 time:  6.540634632110596
166 time:  6.572633743286133
167 time:  6.605633735656738
168 time:  6.639639377593994
169 time:  6.67363429069519
170 time:  6.7066330909729
171 time:  6.736635684967041
172 time:  6.769636392593384
173 time:  6.801632881164551
174 time:  6.835632801055908
175 time:  6.867634057998657
176 time:  6.899636268615723
177 time:  6.93363618850708
178 time:  6.966633081436157
179 time:  6.999640226364136
180 time:  7.030665159225464
181 time:  7.061632394790649
182 time:  7.092637300491333
183 time:  7.122634649276733
184 time:  7.151636362075806
185 time:  7.1806321144104
186 time:  7.21164083480835
187 time:  7.244637727737427
188 time:  7.276642560958862
189 time:  7.306660890579224
190 time:  7.339632987976074
191 time:  7.369633674621582
192 time:  7.399653911590576
193 time:  7.43217921257019
194 time:  7.463184833526611
195 time:  7.494180917739868
196 time:  7.523183584213257
197 time:  7.553179979324341
198 time:  7.582208871841431
199 time:  7.61518406867981
200 time:  7.644188642501831
201 time:  7.674182891845703
202 time:  7.704183340072632
203 time:  7.73318076133728
204 time:  7.7651824951171875
205 time:  7.796178817749023
206 time:  7.826218366622925
207 time:  7.855186939239502
208 time:  7.887182950973511
209 time:  7.9191789627075195
210 time:  7.950189113616943
211 time:  7.979245662689209
212 time:  8.008226871490479
213 time:  8.040181875228882
214 time:  8.069237232208252
215 time:  8.103188276290894
216 time:  8.139707565307617
217 time:  8.177712202072144
218 time:  8.213711977005005
219 time:  8.249709606170654
220 time:  8.28270959854126
221 time:  8.320709943771362
222 time:  8.352719783782959
223 time:  8.389708757400513
224 time:  8.421707391738892
225 time:  8.454731225967407
226 time:  8.48570966720581
227 time:  8.515729665756226
228 time:  8.550708055496216
229 time:  8.580706596374512
230 time:  8.61270785331726
231 time:  8.64473009109497
232 time:  8.676709175109863
233 time:  8.706707954406738
234 time:  8.74070954322815
235 time:  8.77273154258728
236 time:  8.802709817886353
237 time:  8.834707975387573
238 time:  8.866709232330322
239 time:  8.896708488464355
240 time:  8.927706956863403
241 time:  8.959715366363525
242 time:  8.991709470748901
243 time:  9.02473759651184
244 time:  9.0577392578125
245 time:  9.089718580245972
246 time:  9.121753931045532
247 time:  9.156758069992065
248 time:  9.18575668334961
249 time:  9.21975827217102
250 time:  9.249759912490845
251 time:  9.28075361251831
252 time:  9.311750650405884
253 time:  9.34175157546997
254 time:  9.374766111373901
255 time:  9.404765605926514
256 time:  9.434756755828857
257 time:  9.46575403213501
258 time:  9.498756170272827
259 time:  9.531751155853271
260 time:  9.567752361297607
261 time:  9.599751949310303
262 time:  9.630752801895142
263 time:  9.659753322601318
264 time:  9.690781831741333
265 time:  9.721758365631104
266 time:  9.752787113189697
267 time:  9.783798933029175
268 time:  9.818784952163696
269 time:  9.851795673370361
270 time:  9.883819341659546
271 time:  9.914787292480469
272 time:  9.94630742073059
273 time:  9.977309703826904
274 time:  10.01032543182373
275 time:  10.041320085525513

Process finished with exit code 0


'''