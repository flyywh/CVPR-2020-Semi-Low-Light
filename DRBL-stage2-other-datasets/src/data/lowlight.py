import os
from data import srdata
import glob

class LowLight(srdata.SRData):
    def __init__(self, args, name='LowLight', train=True, benchmark=False):
        super(LowLight, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, dir_data):
        super(LowLight, self)._set_filesystem(dir_data)
        self.apath = '/data1/yangwenhan/datasets/'
        print(self.apath)

        self.dir_hr   = os.path.join(self.apath, 'Normal')
        self.dir_lr   = os.path.join(self.apath, 'Low_degraded')
        self.dir_lrr = os.path.join(self.apath, 'Low_real_test_2_rs')
        self.dir_hq = os.path.join(self.apath, 'AVA_good_2')

    def _scan(self):
        names_hr, names_lr, names_lrr, names_hq = super(LowLight, self)._scan()
        names_hr   = names_hr[self.begin - 1:self.end]
        names_lr   = names_lr[self.begin - 1:self.end]
        names_lrr  = names_lrr[self.begin - 1:self.end]
        names_hq   = names_hq[self.begin - 1:self.end]

        return names_hr, names_lr, names_lrr, names_hq
