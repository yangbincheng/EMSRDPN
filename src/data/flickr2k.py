import os
from data import srdata

class Flickr2K(srdata.SRData):
    def __init__(self, args, name='Flickr2K', train=True, benchmark=False):
        #print("args.data_range: ", args.data_range);
        #print("train: ", train);
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]
        #print("data_range: ", data_range)

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(Flickr2K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        #print("This is the Flickr2K class!")
        names_hr, names_lr = super(Flickr2K, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        ##print("self.begin:", self.begin);
        ##print("self.end:", self.end);
        ##print("names_hr:", names_hr);
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]
        #print(self.dir_hr);
        #print(self.dir_lr);
 
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(Flickr2K, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'Flickr2K_HR')
        self.dir_lr = os.path.join(self.apath, 'Flickr2K_LR_bicubic')
        if self.input_large: self.dir_lr += 'L'

