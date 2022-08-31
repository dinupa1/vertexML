import numpy as np
import torch

from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.ROOT)

# split the data to train, validation and test
def RandomSplit(dataset, valid_len, test_len, train_batch, valid_batch=16, test_batch=16):
    valid_size = int(len(dataset)* valid_len)
    test_size = int(len(dataset)* test_len)
    train_size = len(dataset) - valid_size - test_size

    train_data, valid_data, test_data = random_split(dataset, (train_size, valid_size, test_size))

    train_set = DataLoader(train_data, batch_size=train_batch, shuffle=True)
    valid_set = DataLoader(valid_data, batch_size=valid_batch, shuffle=True)
    test_set = DataLoader(test_data, batch_size=test_batch, shuffle=True)

    data = {'train': train_set, 'valid': valid_set, 'test': test_set}

    return data

# train function
def Train(model, train_loader, optmizer, epoch):
    model.train()
    losses, N = [], len(train_loader)
    for batch_idx, data in enumerate(train_loader):
        if(len(data.x)==0): continue
        optmizer.zero_grad()
        output = model(data)
        y, output = data.y, output.squeeze(1)
        loss = F.mse_loss(output, y)
        loss.backward()
        optmizer.step()
        losses.append(loss.item())
    print('epoch : ', epoch)
    print('loss : ', np.nanmean(losses))
    return np.nanmean(losses)

# validation function
def Validation(model, valid_loader, epoch):
    model.eval()
    losses, N = [], len(valid_loader)
    for batch_idx, data in enumerate(valid_loader):
        torch.no_grad()
        if(len(data.x)==0): continue
        output = model(data)
        y, output = data.y, output.squeeze()
        loss = F.mse_loss(output, y)
        losses.append(loss.item())
    print('epoch : ', epoch)
    print('loss : ', np.nanmean(losses))
    return np.nanmean(losses)

# make predictions
def Predict(model, data, batch_size = 16):
    target = {'true': {'charge': [], 'vtx': [], 'vty': [], 'vtz': [], 'vpx': [], 'vpy': [], 'vpz': []},
              'prediction': {'charge': [], 'vtx': [], 'vty': [], 'vtz': [], 'vpx': [], 'vpy': [], 'vpz': []}}


    for idx, batch in enumerate(data):
        pred_list = []
        output = model(batch).squeeze()
        for z in range(batch_size):
            pred_list.append(output[batch.batch == z])
        batch_list = batch.to_data_list()
        for j, event in enumerate(batch_list):
            # target['true']['charge'].append(event.y[0].detach().numpy())
            target['true']['vtx'].append(event.y[1].detach().numpy())
            target['true']['vty'].append(event.y[2].detach().numpy())
            target['true']['vtz'].append(event.y[3].detach().numpy())
            # target['true']['vpx'].append(event.y[4].detach().numpy())
            # target['true']['vpy'].append(event.y[5].detach().numpy())
            # target['true']['vpz'].append(event.y[6].detach().numpy())
            # target['prediction']['charge'].append(pred_list[j][0].detach().numpy())
            target['prediction']['vtx'].append(pred_list[j][1].detach().numpy())
            target['prediction']['vty'].append(pred_list[j][2].detach().numpy())
            target['prediction']['vtz'].append(pred_list[j][3].detach().numpy())
            # target['prediction']['vpx'].append(pred_list[j][4].detach().numpy())
            # target['prediction']['vpy'].append(pred_list[j][5].detach().numpy())
            # target['prediction']['vpz'].append(pred_list[j][6].detach().numpy())

    # print(len(target['true']['vtx']), ',', len(target['true']['vtx']))

    return target

# plot the loss
def plot_loss(epochs, loss):
    f, ax = plt.subplots()
    ax.plot(epochs, loss['train'], label='training loss')
    ax.plot(epochs, loss['valid'], label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('mse loss')
    # plt.title('loss per epoch')
    plt.legend()
    # plt.savefig('loss.png')
    plt.show()

# plot the prediction
def plot_predition(data):
    # plot charge
    f, ax = plt.subplots()
    ax.hist(np.array(data['true']['charge']), bins=3, range=(-1.5, 1.5), histtype='step', label='true')
    ax.hist(np.array(data['prediction']['charge']), bins=3, range=(-1.5, 1.5), histtype='step', label='prediction')
    plt.xlabel('q')
    plt.ylabel('counts')
    # plt.title('charge')
    plt.legend()
    # plt.savefig('charge.png')
    plt.show()

    # plot vtx
    f, ax = plt.subplots()
    ax.hist(np.array(data['true']['vtx']), bins=20, range=(-5.0, 5.0), histtype='step', label='true')
    ax.hist(np.array(data['prediction']['vtx']), bins=20, range=(-5.0, 5.0), histtype='step', label='prediction')
    plt.xlabel('x [cm]')
    plt.ylabel('counts')
    # plt.title('x')
    plt.legend()
    # plt.savefig('vtx.png')
    plt.show()

    # plot vty
    f, ax = plt.subplots()
    ax.hist(np.array(data['true']['vty']), bins=20, range=(-5.0, 5.0), histtype='step', label='true')
    ax.hist(np.array(data['prediction']['vty']), bins=20, range=(-5.0, 5.0), histtype='step', label='prediction')
    plt.xlabel('y [cm]')
    plt.ylabel('counts')
    plt.legend()
    # plt.savefig('vty.png')
    plt.show()

    # plot vtz
    f, ax = plt.subplots()
    ax.hist(np.array(data['true']['vtz']), bins=20, range=(-800.0, 200.0), histtype='step', label='true')
    ax.hist(np.array(data['prediction']['vtz']), bins=20, range=(-800.0, 200.0), histtype='step', label='prediction')
    plt.xlabel('z [cm]')
    plt.ylabel('counts')
    plt.legend()
    # plt.savefig('vtz.png')
    plt.show()

    # # plot vpx
    # f, ax = plt.subplots()
    # ax.hist(np.array(data['true']['vpx']), bins=20, range=(-5.0, 5.0), histtype='step', label='true')
    # ax.hist(np.array(data['prediction']['vpx']), bins=20, range=(-5.0, 5.0), histtype='step', label='prediction')
    # plt.xlabel('px [GeV/c]')
    # plt.ylabel('counts')
    # plt.legend()
    # plt.savefig('vpx.png')
    # # plot vpy
    # f, ax = plt.subplots()
    # ax.hist(np.array(data['true']['vpy']), bins=20, range=(-5.0, 5.0), histtype='step', label='true')
    # ax.hist(np.array(data['prediction']['vpy']), bins=20, range=(-5.0, 5.0), histtype='step', label='prediction')
    # plt.xlabel('py [GeV/c]')
    # plt.ylabel('counts')
    # plt.legend()
    # plt.savefig('vpy.png')
    # # plot vpz
    # f, ax = plt.subplots()
    # ax.hist(np.array(data['true']['vpz']), bins=20, range=(10.0, 100.0), histtype='step', label='true')
    # ax.hist(np.array(data['prediction']['vpz']), bins=20, range=(10.0, 100.0), histtype='step', label='prediction')
    # plt.xlabel('pz [GeV/c]')
    # plt.ylabel('counts')
    # plt.legend()
    # plt.savefig('vpz.png')
    # plot res_vtx
    f, ax = plt.subplots()
    res_vtx = np.array(data['true']['vtx']) - np.array(data['prediction']['vtx'])
    ax.hist(res_vtx, bins=20, range=(-5.0, 5.0), histtype='step')
    plt.xlabel(r'$\Delta x$ [cm]')
    plt.ylabel('counts')
    # plt.legend()
    plt.savefig('res_vtx.png')
    # plot res_vty
    f, ax = plt.subplots()
    res_vty = np.array(data['true']['vty']) - np.array(data['prediction']['vty'])
    ax.hist(res_vty, bins=20, range=(-5.0, 5.0), histtype='step')
    plt.xlabel(r'$\Delta y$ [cm]')
    plt.ylabel('counts')
    # plt.legend()
    plt.savefig('res_vty.png')
    # plot res_vtz
    f, ax = plt.subplots()
    res_vtz = np.array(data['true']['vtz']) - np.array(data['prediction']['vtz'])
    ax.hist(res_vtz, bins=20, range=(-800.0, 200.0), histtype='step')
    plt.xlabel(r'$\Delta z$ [cm]')
    plt.ylabel('counts')
    # plt.legend()
    plt.savefig('res_vtz.png')
    # # plot res_vpx
    # f, ax = plt.subplots()
    # res_vpx = np.array(data['true']['vpx']) - np.array(data['prediction']['vpx'])
    # ax.hist(res_vpx, bins=20, range=(-5.0, 5.0), histtype='step')
    # plt.xlabel(r'$\Delta px$ [GeV/c]')
    # plt.ylabel('counts')
    # # plt.legend()
    # plt.savefig('res_vpx.png')
    # # plot res_vpy
    # f, ax = plt.subplots()
    # res_vpy = np.array(data['true']['vpy']) - np.array(data['prediction']['vpy'])
    # ax.hist(res_vpy, bins=20, range=(-5.0, 5.0), histtype='step')
    # plt.xlabel(r'$\Delta py$ [GeV/c]')
    # plt.ylabel('counts')
    # # plt.legend()
    # plt.savefig('res_vpy.png')
    # # plot res_vpz
    # f, ax = plt.subplots()
    # res_vpx = np.array(data['true']['vpz']) - np.array(data['prediction']['vpz'])
    # ax.hist(res_vpx, bins=20, range=(-10.0, 10.0), histtype='step')
    # plt.xlabel(r'$\Delta pz$ [GeV/c]')
    # plt.ylabel('counts')
    # # plt.legend()
    # plt.savefig('res_vpz.png')