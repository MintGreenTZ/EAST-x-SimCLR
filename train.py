import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from torchtools.optim import Lookahead, RAdam, PlainRAdam, AdamW, Ralamb, RangerLars
from torchvision import transforms
from dataset import custom_dataset
from PIL import Image
from model import EAST
from loss import Loss
from gaussian_blur import GaussianBlur
from eval import eval_model, model_name, test_img_path, submit_path
import os
import time
import numpy as np
# import glovar

best_hmean = -2e15
best_epoch = 0

def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval):
    # global best_epoch
    file_num = len(os.listdir(train_img_path))
    trainset = custom_dataset(train_img_path, train_gt_path)
    train_loader = data.DataLoader(trainset, batch_size=batch_size, \
                                   shuffle=True, num_workers=num_workers, drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = Loss(device, batch_size, temperature=0.5, use_cosine_similarity=True)
    model = EAST()
    data_parallel = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True
    model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = RangerLars(model.parameters())
    # optimizer = Lookahead(base_optimizer=optimizer, k=5, alpha=0.5) #lookahead
    #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter // 2], gamma=0.1)

    # checkpiont 断点训练
    # checkpoint = torch.load('/home/weiran/ICDAR_2015/simclr15_ck_pths/checkpoint')
    # model.module.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # epoch_iter =epoch_iter - checkpoint['epoch']
    # scheduler.load_state_dict(checkpoint['scheduler'])



    # for i, img_file in enumerate(img_files):
    # 	im = Image.open(img_file)
    # 	print(img2tensor(im))
    # 	data_transforms(im)
    # 	print(img2tensor(im))
    # 	print(data_transforms(im))
    # 	data_transforms(im)
    # 	print(img2tensor(im))
    # 	print(data_transforms(im))
    # 	print(data_transforms(im).size())
    # 	if i == 2:
    # 		break
    #
    # print("img finished")

    for epoch in range(epoch_iter):
        model.train()
        #scheduler.step()
        epoch_loss = 0
        epoch_time = time.time()
        for i, (img1, gt_score1, gt_geo1, ignored_map1, img2, gt_score2, gt_geo2, ignored_map2) in enumerate(train_loader):
            # print(img.size())
            # print(gt_score.size())
            # print(gt_geo.size())
            # print(ignored_map.size())

            start_time = time.time()

            img1, gt_score1, gt_geo1, ignored_map1 = img1.to(device), gt_score1.to(device), gt_geo1.to(
                device), ignored_map1.to(device)
            pred_score1, pred_geo1, merged_feature1= model(img1)
            # print(pred_score1.size()) 	# torch.Size([24, 1, 128, 128])
            # print(pred_geo1.size())		# torch.Size([24, 5, 128, 128])

            img2, gt_score2, gt_geo2, ignored_map2 = img2.to(device), gt_score2.to(device), gt_geo2.to(
                device), ignored_map2.to(device)
            pred_score2, pred_geo2, merged_feature2 = model(img2)

            loss = criterion(gt_score1, pred_score1, gt_geo1, pred_geo1, ignored_map1, merged_feature1,\
							 gt_score2, pred_score2, gt_geo2, pred_geo2, ignored_map2, merged_feature2, epoch)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format( \
                epoch + 1, epoch_iter, i + 1, int(file_num / batch_size), time.time() - start_time, loss.item()))

        print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss / int(file_num / batch_size),
                                                                  time.time() - epoch_time))
        print(time.asctime(time.localtime(time.time())))
        print('=' * 50)
        if epoch > 399 and (epoch + 1) % interval == 0:
            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            torch.save(state_dict, os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch + 1)))
            state = {'model': model.module.state_dict() if data_parallel else model.state_dict(), \
                     'optimizer': optimizer.state_dict(), 'epoch': epoch}#, \
                     #'scheduler': scheduler.state_dict()}
            torch.save(state, checkpoint_path)

        global best_hmean
        global best_epoch
        if epoch >= 400:
            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            # print("begin eval_model")
            torch.save(state_dict, os.path.join(pths_path, 'current.pth'))
            hmean = eval_model(os.path.join(pths_path, 'current.pth'), test_img_path, submit_path)
            # print("end eval_model")
            if (hmean > best_hmean):
                best_hmean = hmean
                best_epoch = epoch
                torch.save(state_dict, os.path.join(pths_path, 'best.pth'))
                print("*** The hmean of best model is updated to %.10f ***" % best_hmean)

        print("—— Best epoch ever since is the %d-th epoch" % best_epoch)


if __name__ == '__main__':
    train_img_path = os.path.abspath('../ICDAR_2015/train_img')
    train_gt_path = os.path.abspath('../ICDAR_2015/train_gt')
    pths_path = '../ICDAR_2015/simclr15_pths/'
    checkpoint_path = '/home/weiran/ICDAR_2015/simclr15_ck_pths/checkpoint'
    batch_size = 10
    lr = 1e-3
    num_workers = 4
    epoch_iter = 600
    save_interval = 20
    s = 1
    train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, save_interval)
