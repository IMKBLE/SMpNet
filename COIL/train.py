import torch
from data.dataloader import data_loader_train
from models.network import Networks
import models.metrics as metrics
import numpy as np
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize
from op import loss_function


def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp


def post_proC(C, K, d=6, alpha=8):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5 * (C + C.T)
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L


def graph_loss(Z, S):
    S = 0.5 * (S.permute(1, 0) + S)
    D = torch.diag(torch.sum(S, 1))
    L = D - S
    return 2 * torch.trace(torch.matmul(torch.matmul(Z.permute(1, 0), L), Z))


train_num = 1440
in_size = 128
out_size = 32
out_num = 30720
k_num = 20

data_0 = sio.loadmat('rand/coil_edge_ori_n.mat')
data_dict = dict(data_0)
data0 = data_dict['groundtruth'].T
label_true = np.zeros(train_num)
for i in range(train_num):
    label_true[i] = data0[i]

reg2 = 1.0 * 10 ** (k_num / 10.0 - 3.0)

n_epochs2 = 5001
ACC_coil = np.zeros((1, 1 + int(n_epochs2 / 10)))
NMI_coil = np.zeros((1, 1 + int(n_epochs2 / 10)))
LOSS_coil = np.zeros((1, 1 + int(n_epochs2 / 10)))
model = Networks()
# model.load_state_dict(torch.load('./models/AE1.pth'))

# criterion = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=0.0)
# n_epochs = 1500
# for epoch in range(n_epochs):
#     for data in data_loader_train:
#         train_imga, train_imgb = data
#         input1 = train_imga.view(train_num, 3, in_size, in_size)
#         input2 = train_imgb.view(train_num, 1, in_size, in_size)
#         loss = 0.5 * criterion(output1, input1) + 0.5 * criterion(output2, input2)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     if epoch % 100 == 0:
#         print("Epoch {}/{}".format(epoch, n_epochs))
#         print("Loss is:{:.4f}".format(loss.item()))
# torch.save(model.state_dict(), './models/AE1.pth')

PREDS_coil = np.zeros((int(n_epochs2 / 10) + 1, train_num))
print("step2")
criterion2 = torch.nn.MSELoss(reduction='sum')
optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=0.0)
n_epochs2 = 5001
maxacc = 0
acc_nmi = 0
t0 = torch.zeros(train_num,train_num,requires_grad=True)
for epoch in range(n_epochs2):
    for data in data_loader_train:
        train_imga, train_imgb = data
        input1 = train_imga.view(train_num, 1, in_size, in_size)
        input2 = train_imgb.view(train_num, 1, in_size, in_size)
#        input3 = train_imgc.view(train_num, 1, in_size, in_size)
        z11, z12, z13, zz11, zz12, zz13, z21, z22, z23, zz21, zz22, zz23, output1, output2, coef, coefd11, coefd12, coefd13,coefd21, coefd22, coefd23 = model.forward2(input1, input2)
        loss_re_c = 0.1 * torch.norm(coef, p='nuc')
        loss_re_d = 0.0001 * (criterion2(coefd11, t0) + criterion2(coefd12, t0) + criterion2(coefd13,t0) + criterion2(coefd21, t0) + criterion2(coefd22,t0) + criterion2(coefd23, t0))
        loss_e = reg2 * (0.4 * criterion2(z11, zz11) + 0.4 * criterion2(z12, zz12) + 0.4 * criterion2(z13, zz13) + 0.1 * criterion2(z21, zz21) + 0.1 * criterion2(z22, zz22) + 0.1 * criterion2(z23, zz23))
        loss_r = 0.8 * criterion2(output1, input1) + 0.2 * criterion2(output2, input2) 
        # loss_g1 = graph_loss(z1, coef)
        # loss_g2 = graph_loss(z2, coef)
        # norm = coef.shape[0] * coef.shape[0] / float((coef.shape[0] * coef.shape[0] - coef.sum()) * 2)
        # loss = 1 * loss_r + 0.1 * loss_re + reg2 * loss_e + loss_g1 / train_num + loss_g2 / train_num
        loss = 1 * loss_r + loss_re_c + loss_re_d + loss_e
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        if epoch % 10 == 0:
            print("Epoch {}/{}".format(epoch, n_epochs2))
            print("Loss is:{:.4f}".format(loss.item()))
            print("Lossr is:{:.4f}".format(loss_r.item()))
            print("Losse is:{:.4f}".format(loss_e.item()))
            print("Lossre_c is:{:.4f}".format(loss_re_c.item()))
            print("Lossre_d is:{:.4f}".format(loss_re_d.item()))
            coef = coef.cpu().detach().numpy()
            # coefd11 = coefd11.cpu().detach().numpy()
            # coefd12 = coefd12.cpu().detach().numpy()
            # coefd13 = coefd13.cpu().detach().numpy()
            # coefd21 = coefd21.cpu().detach().numpy()
            # coefd22 = coefd22.cpu().detach().numpy()
            # coefd23 = coefd23.cpu().detach().numpy()
            # coefd31 = coefd31.cpu().detach().numpy()
            # coefd32 = coefd32.cpu().detach().numpy()
            # coefd33 = coefd33.cpu().detach().numpy()

            alpha = max(0.4 - (k_num - 1) / 10 * 0.1, 0.1)
            # commonZ = thrC(abs(coef + (coefd11 + coefd12 + coefd13 + coefd21 + coefd22 + coefd23 + coefd31 + coefd32 + coefd33)/9) + abs(coef + (coefd11 + coefd12 + coefd13 + coefd21 + coefd22 + coefd23 + coefd31 + coefd32 + coefd33)/9).T, alpha)
            commonZ = thrC(abs(coef) + abs(coef).T, alpha)
            preds, _ = post_proC(commonZ, k_num)
            acc = metrics.acc(label_true, preds)
            nmi = metrics.nmi(label_true, preds)
            ACC_coil[0, int(epoch / 10)] = acc
            NMI_coil[0, int(epoch / 10)] = nmi
            LOSS_coil[0, int(epoch / 10)] = loss
            PREDS_coil[int(epoch / 10), :] = preds
            print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f, maxacc: %.4f, acc_nmi: %.4f  <==|'
                  % (acc, nmi, maxacc, acc_nmi))
            if acc > maxacc:
                maxacc, acc_nmi = acc, nmi
            if epoch%100==0:
                Z_path = 'commonZ' + str(epoch)
                sio.savemat(Z_path + '.mat', {'Z': commonZ})
p_path = 'preds_coil'
sio.savemat(p_path + '.mat', {'PREDS_coil': PREDS_coil})
torch.save(model.state_dict(), './models/AE2.pth')
sio.savemat('NMI_coil' + '.mat', {'NMI_coil': NMI_coil})
sio.savemat('ACC_coil' + '.mat', {'ACC_coil': ACC_coil})
sio.savemat('LOSS_coil' + '.mat', {'LOSS_coil': LOSS_coil})
