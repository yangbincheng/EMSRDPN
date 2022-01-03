import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

#torch.set_printoptions(profile="full")
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if args.data_test == 'video':
    from videotester import VideoTester
    model = model.Model(args, checkpoint)
    t = VideoTester(args, model, checkpoint)
    t.test()
else:
    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        checkpoint.write_log("total params: {}, total trainable params: {}".format(pytorch_total_params, pytorch_total_trainable_params))
 
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model, loss, checkpoint)

        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()

