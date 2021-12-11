import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


train_num = 1440
out_size = 32
out_num = 30720
k_num = 20

'''
class Networks(nn.Module):
    def __init__(self):
        super(Networks, self).__init__()
        self.dc = InnerProductDecoder(dropout=0.0, act=lambda x: x)

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(20, 30, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(30, 20, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(20, 10, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 3, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(20, 30, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(30, 20, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(20, 10, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 1, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )

        self.weight = nn.Parameter(1.0e-4 * torch.ones(trian_num, trian_num))

    def forward(self, input1, input2):
        output1 = self.encoder1(input1)
        output1 = self.decoder1(output1)
        output2 = self.encoder2(input2)
        output2 = self.decoder2(output2)

        return output1, output2

    def forward2(self, input1, input2):
        coef = self.weight - torch.diag(torch.diag(self.weight))
        z1 = self.encoder1(input1)
        z1 = z1.view(trian_num, out_num)
        zcoef1 = torch.matmul(coef, z1)
        output1 = zcoef1.view(trian_num, 30, out_size, out_size)
        output1 = self.decoder1(output1)
        z2 = self.encoder2(input2)
        z2 = z2.view(trian_num, out_num)
        zcoef2 = torch.matmul(coef, z2)
        output2 = zcoef2.view(trian_num, 30, out_size, out_size)
        output2 = self.decoder2(output2)

        return z1, zcoef1, output1, coef, z2, zcoef2, output2, self.dc(coef)
'''


class Networks(nn.Module):
    def __init__(self):
        super(Networks, self).__init__()
        self.encoder11 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )
        self.encoder12 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )
        self.encoder13 = nn.Sequential(
            nn.Conv2d(20, 30, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )

        self.decoder11 = nn.Sequential(
            nn.ConvTranspose2d(30, 20, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )
        self.decoder12 = nn.Sequential(
            nn.ConvTranspose2d(20, 10, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )
        self.decoder13 = nn.Sequential(
            nn.ConvTranspose2d(10, 1, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )

        self.encoder21 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )
        self.encoder22 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )
        self.encoder23 = nn.Sequential(
            nn.Conv2d(20, 30, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )

        self.decoder21 = nn.Sequential(
            nn.ConvTranspose2d(30, 20, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )
        self.decoder22 = nn.Sequential(
            nn.ConvTranspose2d(20, 10, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )
        self.decoder23 = nn.Sequential(
            nn.ConvTranspose2d(10, 1, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )
        

        self.weight = nn.Parameter(1.0e-4 * torch.ones(train_num, train_num))

        self.weightd11 = nn.Parameter(1.0e-4 * torch.ones(train_num, train_num))
        self.weightd12 = nn.Parameter(1.0e-4 * torch.ones(train_num, train_num))
        self.weightd13 = nn.Parameter(1.0e-4 * torch.ones(train_num, train_num))

        self.weightd21 = nn.Parameter(1.0e-4 * torch.ones(train_num, train_num))
        self.weightd22 = nn.Parameter(1.0e-4 * torch.ones(train_num, train_num))
        self.weightd23 = nn.Parameter(1.0e-4 * torch.ones(train_num, train_num))



    def forward(self, input1, input2):
        output11 = self.encoder11(input1)
        output12 = self.encoder12(output11)
        output13 = self.encoder13(output12)

        output12 = self.decoder11(output13)
        output11 = self.decoder12(output12)
        output1 = self.decoder13(output11)

        output21 = self.encoder21(input2)
        output22 = self.encoder22(output21)
        output23 = self.encoder23(output22)

        output22 = self.decoder21(output23)
        output21 = self.decoder22(output22)
        output2 = self.decoder23(output21)



        return output1, output2

    def forward2(self, input1, input2):
        coef = self.weight - torch.diag(torch.diag(self.weight))

        coefd11 = self.weightd11 - torch.diag(torch.diag(self.weightd11))
        coefd12 = self.weightd12 - torch.diag(torch.diag(self.weightd12))
        coefd13 = self.weightd13 - torch.diag(torch.diag(self.weightd13))
        coefd21 = self.weightd21 - torch.diag(torch.diag(self.weightd21))
        coefd22 = self.weightd22 - torch.diag(torch.diag(self.weightd22))
        coefd23 = self.weightd23 - torch.diag(torch.diag(self.weightd23))


        z1 = self.encoder11(input1)
        z11 = z1.view(train_num, -1)
        z2 = self.encoder12(z1)
        z12 = z2.view(train_num, -1)
        z3 = self.encoder13(z2)
        z13 = z3.view(train_num, out_num)

        zz13 = torch.matmul(coef + coefd13, z13)
        output13 = zz13.view(train_num, 30, out_size, out_size)
        output12 = self.decoder11(output13)
        zz12 = torch.matmul(coef + coefd12, z12)
        zzz12 = zz12.view(train_num, 20, 64, 64)
        output11 = self.decoder12(output12+zzz12)
        zz11 = torch.matmul(coef + coefd11, z11)
        zzz11 = zz11.view(train_num, 10, 64, 64)
        output1 = self.decoder13(output11+zzz11)

        z2 = self.encoder21(input2)
        z21 = z2.view(train_num, -1)
        z2 = self.encoder22(z2)
        z22 = z2.view(train_num, -1)
        z2 = self.encoder23(z2)
        z23 = z2.view(train_num, out_num)

        zz23 = torch.matmul(coef + coefd23, z23)
        output23 = zz23.view(train_num, 30, out_size, out_size)
        output22 = self.decoder21(output23)
        zz22 = torch.matmul(coef + coefd22, z22)
        zzz22 = zz22.view(train_num, 20, 64, 64)
        output21 = self.decoder22(output22+zzz22)
        zz21 = torch.matmul(coef + coefd21, z21)
        zzz21 = zz21.view(train_num, 10, 64, 64)
        output2 = self.decoder23(output21+zzz21)



        return z11, z12, z13, zz11, zz12, zz13, z21, z22, z23, zz21, zz22, zz23, output1, output2, coef, coefd11, coefd12, coefd13,\
               coefd21, coefd22, coefd23







