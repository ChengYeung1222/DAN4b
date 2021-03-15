import os
from custom_data_io import custom_dset
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import Models as models

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Testing settings
batch_size = 1
no_cuda = False
# seed = 8
target_list = "./depth_500/DYGZ_deep_500.csv"
target_name = 'unknown zone'
ckpt_model = './ckpt_reasonable/model_epoch22.pth'  # todo

cuda = not no_cuda and torch.cuda.is_available()

target_dataset = custom_dset(txt_path=target_list, nx=227, nz=227, labeled=True)

target_train_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True,
                                 drop_last=False)

len_target_dataset = len(target_dataset)
len_target_loader = len(target_train_loader)


def load_ckpt(model):
    model.load_state_dict(torch.load(ckpt_model))
    return model


def predict(model):
    model.eval()
    iter_target = iter(target_train_loader)
    num_iter_target = len_target_dataset

    output_file = open("prediction_500_wommd.txt", 'w')
    output_file.write('Sample, ' + 'Label, ' + 'Prediction' + '\n')

    for i in range(0, num_iter_target):
        data, label = next(iter_target)
        data, label = data.float(), label.long()
        if cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        output, _ = model(data, data)
        pred_label = output.data.max(1)[1].cpu().numpy()[0]
        score=output.data.cpu().numpy()[:,1]
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
