import torch
from utils import def_para
from network.STTC_L0_NN import STTC_L0_NN
from utils import metrics
from utils import Loss
import os
import time
import gc
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    para, X, train_data, test_data, scale = def_para()
    model = STTC_L0_NN(para).to(device)
    G_m_pre = [None] * para['dim']
    Gtr_pre = [None] * para['dim']
    G_cat_pre = [None] * para['dim']
    X_rec_pre = [None] * para['dim']
    R_m_pre = [None] * para['dim']
    optimizer = torch.optim.Adam(model.parameters(), lr=para['lr'])
    criterion = Loss().to(device)
    print("Start training...")
    for epoch in range(20):
        for i in range(para['training set size']):
            G_m_pre = train_data['G_m'][f'G_m{i}']
            Gtr_pre = train_data['Gtr'][f'Gtr{i}']
            G_cat_pre = train_data['G_cat'][f'G_cat{i}']
            X_rec_pre = train_data['X_rec'][f'X_rec{i}']
            A_pre = train_data['A'][i]
            R_m_pre = train_data['R_m'][f'R_m{i}']
            Omega = train_data['Omega'][i]
            scale_pre = scale
            optimizer.zero_grad()
            iter_time_start = time.time()
            G_m, Gtr, G_cat, X_rec, A, R_m, scale = model(X, G_m_pre, Gtr_pre, G_cat_pre, X_rec_pre, A_pre, R_m_pre, Omega, scale_pre)
            loss_normal = criterion(X_rec * (1 - Omega), X * (1 - Omega))
            loss_normal.backward(retain_graph=True)
            optimizer.step()
            iter_time_end = time.time()
            rmae, rmse, mae = metrics(X, X_rec, (1 - Omega))
            iter_time = iter_time_end - iter_time_start
            print(f'\n [epoch:{epoch + 1}][{i + 1}/{para["training set size"]}]   MAE:{mae.item():.4f} RMSE:{rmse.item():.4f} Time：{iter_time:.4f}s')
            train_data['G_m'][f'G_m{i}'] = G_m
            train_data['Gtr'][f'Gtr{i}'] = Gtr
            train_data['G_cat'][f'G_cat{i}'] = G_cat
            train_data['X_rec'][f'X_rec{i}'] = X_rec
            train_data['A'][i] = A
            train_data['R_m'][f'R_m{i}'] = R_m
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join('network', 'model{}.pth'.format(epoch + 1)))
    model.eval()
    print("Training complete")
    print("Start testing...")
    with torch.no_grad():
        scale = 10000
        for epoch in range(20):
            for i in range(para['testing set size']):
                gc.collect()
                G_m_pre = test_data['G_m'][f'G_m{i}']
                Gtr_pre = test_data['Gtr'][f'Gtr{i}']
                G_cat_pre = test_data['G_cat'][f'G_cat{i}']
                X_rec_pre = test_data['X_rec'][f'X_rec{i}']
                A_pre = test_data['A'][i]
                R_m_pre = test_data['R_m'][f'R_m{i}']
                Omega = test_data['Omega'][i]
                scale_pre = scale
                iter_time_start = time.time()
                G_m, Gtr, G_cat, X_rec, A, R_m, scale = model(X, G_m_pre, Gtr_pre, G_cat_pre, X_rec_pre, A_pre, R_m_pre,Omega,scale_pre)
                iter_time_end = time.time()
                rmae, rmse, mae = metrics(X, X_rec, (1 - Omega))
                iter_time = iter_time_end - iter_time_start
                print(f'\n [epoch:{epoch + 1}][{i + 1}/{para["testing set size"]}]   MAE:{mae.item():.4f} RMSE:{rmse.item():.4f} Time：{iter_time:.4f}s')
                test_data['G_m'][f'G_m{i}'] = G_m
                test_data['Gtr'][f'Gtr{i}'] = Gtr
                test_data['G_cat'][f'G_cat{i}'] = G_cat
                test_data['X_rec'][f'X_rec{i}'] = X_rec
                test_data['A'][i] = A
                test_data['R_m'][f'R_m{i}'] = R_m
    print("Testing complete")