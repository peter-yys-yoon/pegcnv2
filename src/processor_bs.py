import time
import torch
import numpy as np
from torch.backends import cudnn
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import src.utils as U
from src.database import NTU
from src.dataprocessor import *
from src.graph import Graph
from src.nets import ST_GCN
from src.mask import Mask
from tqdm import tqdm
from collections import OrderedDict

"""
For training baseline model!


"""
class Processor_BS():
    def __init__(self, args):
        print('Starting preparing ...')
        self.args = args

        # Program setting
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.benchmark = True
        self.device = U.check_gpu(args.gpus)

        # Data Loader Setting
        if args.subset in ['cs', 'cv']:
            num_class = 60
        elif args.subset in ['csub', 'cset']:
            num_class = 120
        else:
            raise ValueError('Do NOT exist this subset: {}'.format(args.subset))
        data_shape = (3, args.max_frame, 25, 2)
        transform = transforms.Compose([
            Data_transform(args.data_transform), 
            Occlusion_part(args.occlusion_part), 
            Occlusion_time(args.occlusion_time), 
            Occlusion_block(args.occlusion_block), 
            Occlusion_rand(args.occlusion_rand, data_shape),
            Jittering_joint(args.jittering_joint, data_shape, sigma=args.sigma),
            Jittering_frame(args.jittering_frame, data_shape, sigma=args.sigma),
        ])
        self.train_loader = DataLoader(NTU('train', args.subset, data_shape, transform=transform),
                                       batch_size=args.batch_size, num_workers=2*len(args.gpus),
                                       pin_memory=True, shuffle=True, drop_last=True)
        self.eval_loader = DataLoader(NTU('eval', args.subset, data_shape, transform=transform),
                                      batch_size=args.batch_size, num_workers=2*len(args.gpus),
                                      pin_memory=True, shuffle=False, drop_last=False)
        if args.data_transform:
            data_shape = (9, args.max_frame, 25, 2)

        # Graph Setting
        graph = Graph(max_hop=args.gcn_kernel_size[1])
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False).to(self.device)

        # Model Setting
        self.model_name = 'baseline_NTU'+args.subset
        self.model = ST_GCN(data_shape, num_class, A, args.drop_prob, args.gcn_kernel_size).to(self.device)
        self.model = nn.DataParallel(self.model)

        # Optimizer Setting
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.learning_rate,
            momentum=0.9, weight_decay=0.0001, nesterov=True)

        # Loss Function Setting
        self.loss_func = nn.CrossEntropyLoss()
        print('Successful!')
        print('Training baseline stream!\n')


    # Getting Model FCN Weights
    def get_weights(self, y=None):
        W = []
        for i in range(self.args.model_stream):
            temp_W = self.model.module.stgcn_stream[i].fcn.weight
            if y is not None:
                temp_W = temp_W[y,:]
            W.append(temp_W.view(temp_W.shape[0], -1))
        return W


    # Learning Rate Adjusting
    def adjust_lr(self, epoch):
        # LR decay
        if epoch in self.args.adjust_lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] /= 10


    # Training
    def train(self, epoch):
        acc, num_sample = 0, 0
        name_desc = tqdm(range(len(self.train_loader)))

        for num, (x, _, y, _) in enumerate(self.train_loader):

            # Using GPU
            x = x.to(self.device)
            y = y.to(self.device)

            # Calculating Output
            out_stream, feature = self.model(x)
            out = out_stream

            # Calculating Loss
            loss = self.loss_func(out, y)

            # Loss Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Calculating Accuracies
            pred = out.max(1, keepdim=True)[1]
            acc += pred.eq(y.view_as(pred)).sum().item()
            num_sample += x.shape[0]

            # print('after loss backward', self.model.module.mask_stream[1].cpu().detach().numpy()[:5])
            # Print Loss
            name_desc.update(1)
            name_desc.set_description('Epoch: {}/{}, Loss: {:.4f}'.format(
                epoch+1, self.args.max_epoch, loss))
            # print('Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}'.format(epoch+1, self.args.max_epoch, num+1, len(self.train_loader), loss))

        name_desc.close()
        return acc / num_sample * 100


    # Testing
    def eval(self):
        with torch.no_grad():
            acc, num_sample = 0, 0
            name_desc = tqdm(range(len(self.eval_loader)))
            for num, (x, _, y, _) in enumerate(self.eval_loader):
                # print('eval, data shape: ', x.shape)

                # Using GPU
                x = x.to(self.device)
                y = y.to(self.device)

                # Calculating Output
                out_stream, _ = self.model(x)
                out = out_stream

                # Calculating Accuracies
                pred = out.max(1, keepdim=True)[1]
                acc += pred.eq(y.view_as(pred)).sum().item()
                num_sample += x.shape[0]

                # Print Progress
                name_desc.update(1)
                # print('Batch: {}/{}'.format(num+1, len(self.eval_loader)))
        name_desc.close()
        return acc / num_sample * 100


    def start(self):
        # Training Start
        start_time = time.time()

        if self.args.evaluate:
            # Loading evaluating model
            print('Loading evaluating model ...')
            checkpoint = U.load_checkpoint(self.args.tag, self.model_name)
            self.model.module.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('Successful!\n')

            # Start evaluating
            print('Starting evaluating ...')
            self.model.module.eval()
            acc = self.eval()
            print('Finish evaluating!')
            print('Best accuracy: {:2.2f}%, Total time:{:.4f}s'.format(acc, time.time()-start_time))

        else:
            # Resuming
            start_epoch, best_acc = 0, 0
            if self.args.resume:
                print('Loading checkpoint ...')
                checkpoint = U.load_checkpoint(self.args.tag)
                self.model.module.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch']
                best_acc = checkpoint['best']
                print('Successful!\n')

            # Start training
            print('Starting training ...')
            self.model.module.train()
            for epoch in range(start_epoch, self.args.max_epoch):

                # Adjusting learning rate
                self.adjust_lr(epoch)

                # Training
                acc = self.train(epoch)
                print('Epoch: {}/{}, Training accuracy: {:2.2f}%, Training time: {:.4f}s\n'.format(
                    epoch+1, self.args.max_epoch, acc, time.time()-start_time))

                # Evaluating
                is_best = False
                if (epoch+1) > self.args.adjust_lr[-1] and (epoch+1) % 2 == 0:
                    print('Evaluating for epoch {} ...'.format(epoch+1))
                    self.model.module.eval()
                    acc = self.eval()
                    print('Epoch: {}/{}, Evaluating accuracy: {:2.2f}%, Evaluating time: {:.4f}s\n'.format(
                        epoch+1, self.args.max_epoch, acc, time.time()-start_time))
                    self.model.module.train()
                    if acc > best_acc:
                        best_acc = acc
                        is_best = True

                # Saving model
                is_parallel = isinstance(self.model, nn.DataParallel)

                if is_parallel:
                    model_dict = self.model.module.state_dict()
                else:
                    model_dict = self.model.state_dict()

                U.save_checkpoint(model_dict, self.optimizer.state_dict(),
                        epoch+1, best_acc, is_best, self.model_name)

            print('Finish training!')
            print('Best accuracy: {:2.2f}%, Total time: {:.4f}s'.format(best_acc, time.time()-start_time))

