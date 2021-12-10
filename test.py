import os
from custom_data_io import custom_dset
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import Models as models

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Testing settings
batch = True
if batch:
    batch_size = 32
else:
    batch_size = 1
no_cuda = False
# seed = 8
target_list = './ssd/ssd_val_s_20c.csv'
# target_list = "./ssd_shallow.csv"  # "./depth_500/DYGZ_deep_500.csv"  ./ssd_adaptation.csv
target_name = 'unknown zone'

ckpt_model = './ckpt_uni_ssd_transfer_heteFalse_break/model_epoch58.pth'
# ckpt_model = './ckpt_uni_ssd_transfer_heteBaseline/model_epoch20.pth'
# ckpt_model = './ckpt_d1500_ssd_parallelpre_210513/model_epoch50.pth'  # todo  ./ckpt_d1500_ssd_parallelmlp_0401/model_epoch_mlp6.pth

cuda = not no_cuda and torch.cuda.is_available()

target_dataset = custom_dset(txt_path=target_list, nx=227, nz=227, labeled=True, test=False)  # todo

target_train_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
                                 drop_last=False)

len_target_dataset = len(target_dataset)
len_target_loader = len(target_train_loader)
# pre_steps = int(len_target_dataset / len_target_loader)
pre_steps = int(len_target_loader)


def load_ckpt(model):
    model.load_state_dict(torch.load(ckpt_model))
    return model


def predict(model):
    model.eval()
    iter_target = iter(target_train_loader)
    num_iter_target = len_target_dataset

    # output_file = open("score_ssd_parallelpre_50_shallow_0524.txt", 'w')  #
    output_file = open("score_ssd_val_heteFalse.csv", 'w')  # prediction_ssd_parallelmlp_6_adapt.txt
    output_file.write('Sample, ' + 'Label, ' + 'Prediction' + '\n')

    if batch:
        for i in range(0, pre_steps):
            data, label, _, _ = next(iter_target)
            data, label = data.float(), label.long()
            if cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            output = model(data, data, 0., 0., 0., 0., False)
            for j in range(batch_size):
                # score = output.data.cpu().numpy()[:, 1]
                score = output[0][:]
                print(str(label.numpy()[j]) + ', ' + str(score[j,0].cpu().data.numpy())+ ', ' + str(score[j,1].cpu().data.numpy()) + '\n')
                output_file.write(str(label.numpy()[j]) + ', ' + str(score[j,0].cpu().data.numpy())+ ', ' + str(score[j,1].cpu().data.numpy()) + '\n')
                # output_file.write(str(label.numpy()[0, j]) + ', ' + str(score[0, j]) + '\n')
    else:
        for i in range(0, num_iter_target):
            data, label, _, _ = next(iter_target)  # todo
            data, label = data.float(), label.long()
            if cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            output, _ = model(data, data, 0., 0., 0., 0., False)
            pred_label = output.data.max(1)[1].cpu().numpy()[0]
            score = output.data.cpu().numpy()[:, 1]
            print(str(i) + ', ' + str(label.numpy()[0]) + ', ' + str(score[0]) + '\n')
            output_file.write(str(i) + ', ' + str(label.numpy()[0]) + ', ' + str(score[0]) + '\n')
    output_file.close()


if __name__ == '__main__':
    # model = models.DANNet(num_classes=2)  # Models.py#todo:
    model = models.DAN_with_Alex(num_classes=2)
    correct = 0
    print(model)
    if cuda:
        model.cuda()

    model = load_ckpt(model)
    predict(model)
