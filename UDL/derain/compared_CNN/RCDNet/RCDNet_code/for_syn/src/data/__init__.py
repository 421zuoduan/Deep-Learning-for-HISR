from importlib import import_module
from data.rainheavytest import TestRealDataset
# from dataloader import MSDataLoader
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import default_collate

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())
            trainset = getattr(module_train, args.data_train)(args)
            self.loader_train = DataLoader(
                # args,
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu
            )

        if args.data_test in ['Set5', 'Set14', 'B100', 'Urban100']:
            module_test = import_module('data.benchmark')
            testset = getattr(module_test, 'Benchmark')(args, train=False)
        else:
            module_test = import_module('data.' + args.data_test.lower())
            # testset_real = TestRealDataset(rgb_dir="./real")#getattr(module_test, "TestRealDataset")(args, rgb_dir="./real")
            testset = getattr(module_test, args.data_test)(args, train=False)

        self.loader_test = DataLoader(
            # args,
            testset,
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu
        )

