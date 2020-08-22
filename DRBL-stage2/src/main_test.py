import torch

import utility
import data
import model
import loss
from option import args
from trainer_test import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)

    my_model = model.Model(args, checkpoint)
    my_model.model.load_state_dict(torch.load('./pretrain/model_s2.pt'))

    args.model = 'RECOMPOSE'
    my_recomp = model.Model(args, checkpoint)
    my_recomp.model.load_state_dict(torch.load('./pretrain/model_R_s2.pt'))

    args.model = 'DISCRIMINATOR'
    my_dis = model.Model(args, checkpoint)
    my_dis.model.load_state_dict(torch.load('./pretrain/model_D_s2.pt'))

    args.n_colors = 3

    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, my_model, my_recomp, my_dis, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

