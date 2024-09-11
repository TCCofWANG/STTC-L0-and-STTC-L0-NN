import torch.nn as nn
import torch
import tensorly as tl
from tensorly.base import unfold
from utils import Ui2U
from utils import inverse
from utils import TCP
from utils import tenspermute
from utils import outliers
import warnings
tl.set_backend('pytorch')
warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class STTC_L0_NN(nn.Module):
    def __init__(self, para):
        super(STTC_L0_NN, self).__init__()
        self.r = para['r']
        self.Data_Size = para['Data_Size']
        self.Block_num = para['Block_num']
        self.tol = para['tol']
        self.inner_tol = para['inner_tol']
        self.lambda_ = para['lambda']
        self.alpha = nn.Parameter(torch.tensor(para['alpha'], dtype=torch.float32))
        self.ni = nn.Parameter(torch.tensor(para['ni'], dtype=torch.float32))
        self.mu = nn.Parameter(torch.tensor(para['mu'], dtype=torch.float32))
        self.W0 = nn.Parameter(torch.tensor(para['W0'], dtype=torch.float32))
        self.B0 = nn.Parameter(torch.tensor(para['B0'], dtype=torch.float32))
        self.W1 = nn.Parameter(torch.tensor(para['W1'], dtype=torch.float32))
        self.B1 = nn.Parameter(torch.tensor(para['B1'], dtype=torch.float32))
        self.W2 = nn.Parameter(torch.tensor(para['W2'], dtype=torch.float32))
        self.B2 = nn.Parameter(torch.tensor(para['B2'], dtype=torch.float32))
        self.epsilon = para['epsilon']

        self.dim = len(self.Data_Size)
        self.train_data_num = para['training set size']
        self.test_data_num = para['testing set size']

        self.Qtr = [None] * len(self.Data_Size)
        self.Ztr = [None] * len(self.Data_Size)
        self.Q_m = [None] * len(self.Data_Size)
        self.U_m = [None] * len(self.Data_Size)
        self.Z_m = [None] * len(self.Data_Size)
        self.Y_m = [None] * len(self.Data_Size)

        self.Updating_G = Updating_G_Layer(self.alpha, self.ni, self.mu)
        self.Updating_Q_0 = Updating_Q_Layer_0(self.W0,self.B0,self.ni)
        self.Updating_Q_1 = Updating_Q_Layer_1(self.W1,self.B1,self.ni)
        self.Updating_Q_2 = Updating_Q_Layer_2(self.W2,self.B2,self.ni)
        self.Updating_Z = Updating_Z_Layer(self.mu)
        self.Updating_U = Updating_U_Layer(self.ni)
        self.Updating_Y = Updating_Y_Layer(self.mu)
        self.Updating_A = Updating_A_Layer(self.epsilon)
        self.Updating_X = Updating_X_Layer(self.Data_Size)


    def forward(self,X,G_m,Gtr,G_cat,X_rec,A,R_m,Omega,scale):

        N_Omega = X_rec - A

        for i in range(self.dim):

            GtrT = [Gtr[j] for j in list(range(i, self.dim)) + list(range(0, i))]
            r_now, In, r_next = GtrT[0].shape
            T = TCP(GtrT[1:])
            T = tenspermute(T, 1)
            C = torch.tensor(unfold(tl.tensor(T), mode=0)).to(device)
            N_m = torch.tensor(unfold(tl.tensor(tenspermute(N_Omega, i)), 0)).to(device)

            self.Z_m[i] = G_m[i]
            self.Y_m[i] = torch.zeros_like(self.Z_m[i])
            self.Q_m[i] = G_m[i]
            self.U_m[i] = torch.zeros_like(self.Q_m[i])

            for j in range(self.Block_num):

                G_m_pre = G_m[i].clone()
                R_m_pre = R_m[i].clone()
                U_m_pre = self.U_m[i].clone()
                Y_m_pre = self.Y_m[i].clone()

                G_m[i] = self.Updating_G(N_m, C, r_now, r_next, G_m_pre, self.Q_m[i], self.Z_m[i], self.U_m[i],self.Y_m[i], i)
                Updating_Q_Method = [self.Updating_Q_0, self.Updating_Q_1, self.Updating_Q_2]
                self.Updating_Q = Updating_Q_Method[i]
                self.Q_m[i], R_m[i] = self.Updating_Q(G_m[i], self.U_m[i], R_m_pre, i)
                self.Z_m[i] = self.Updating_Z(G_m[i], self.Y_m[i], i)
                self.U_m[i] = self.Updating_U(G_m[i], self.Q_m[i], U_m_pre, i)
                self.Y_m[i] = self.Updating_Y(G_m[i], self.Z_m[i], Y_m_pre, i)

                d_GG = torch.sum((G_m[i] - G_m_pre) ** 2) / torch.sum(G_m_pre ** 2)
                d_QG = torch.sum((G_m[i] - self.Q_m[i]) ** 2) / torch.sum(G_m[i] ** 2)
                d_ZG = torch.sqrt(torch.sum((G_m[i] - self.Z_m[i]) ** 2)) / torch.sum(G_m[i] ** 2)
                s_gg = torch.tensor([d_GG, d_QG, d_ZG])

                if torch.max(s_gg) < self.inner_tol:
                    break

            Gtr[i] = G_m[i].reshape(In, r_now, r_next).permute(1, 0, 2)

        R = X_rec - G_cat
        A, scale = self.Updating_A(Omega, R, scale)
        X_rec = self.Updating_X(X, Omega, Gtr)

        return G_m,Gtr,G_cat,X_rec,A,R_m,scale

class Updating_G_Layer(nn.Module):
    def __init__(self,alpha,ni,mu):
        super(Updating_G_Layer,self).__init__()
        self.alpha = alpha
        self.ni = ni
        self.mu = mu

    def forward(self,N_m,C,r_now,r_next,G_pre_i,Q_m_i,Z_m_i,U_m_i,Y_m_i,i):

        A = N_m @ C + 0.5 * self.mu[i].clone() * Z_m_i - 0.5 * Y_m_i + 0.5 * self.ni[i].clone() * Q_m_i - 0.5 * U_m_i + self.alpha[i].clone() * G_pre_i
        B = C.T @ C + (self.alpha[i].clone() + 0.5 * self.mu[i].clone() + 0.5 * self.ni[i].clone()) * torch.eye(r_now * r_next,dtype=torch.float32).to(device)
        G_m_i = inverse(A,B)

        return G_m_i
class Updating_Q_Layer_0(nn.Module):
    def __init__(self,W,B,ni):
        super(Updating_Q_Layer_0,self).__init__()
        self.ni = ni
        self.W = W
        self.B = B

    def forward(self,G_m_i,U_m_i,R_m_i,i):

        A = self.W.clone() * R_m_i + self.B.clone()
        B = self.ni[i].clone() * G_m_i + U_m_i
        Q_m_i = A.clone() @ B
        R_m_i = A.clone()

        return Q_m_i,R_m_i

class Updating_Q_Layer_1(nn.Module):
    def __init__(self,W,B,ni):
        super(Updating_Q_Layer_1,self).__init__()
        self.ni = ni
        self.W = W
        self.B = B

    def forward(self,G_m_i,U_m_i,R_m_i,i):

        A = self.W.clone() * R_m_i + self.B.clone()
        B = self.ni[i].clone() * G_m_i + U_m_i
        Q_m_i = A @ B
        R_m_i = A

        return Q_m_i,R_m_i

class Updating_Q_Layer_2(nn.Module):
    def __init__(self,W,B,ni):
        super(Updating_Q_Layer_2,self).__init__()
        self.ni = ni
        self.W = W
        self.B = B

    def forward(self,G_m_i,U_m_i,R_m_i,i):

        A = self.W.clone() * R_m_i + self.B.clone()
        B = self.ni[i].clone() * G_m_i + U_m_i
        Q_m_i = A @ B
        R_m_i = A

        return Q_m_i,R_m_i



class Updating_Z_Layer(nn.Module):
    def __init__(self,mu):
        super(Updating_Z_Layer,self).__init__()
        self.mu = mu

    def forward(self,G_m_i,Y_m_i,i):

        Z = G_m_i + Y_m_i / self.mu[i].clone()
        Z_m_i = torch.relu(Z)

        return Z_m_i


class Updating_U_Layer(nn.Module):
    def __init__(self,ni):
        super(Updating_U_Layer,self).__init__()
        self.ni = ni

    def forward(self,G_m_i,Q_m_i,U_pre_i,i):

        U_m_i = U_pre_i + self.ni[i].clone() * (G_m_i - Q_m_i)

        return U_m_i


class Updating_Y_Layer(nn.Module):
    def __init__(self,mu):
        super(Updating_Y_Layer,self).__init__()
        self.mu = mu

    def forward(self,G_m_i,Z_m_i,Y_pre_i,i):

        Y_m_i = Y_pre_i + self.mu[i].clone() * (G_m_i - Z_m_i)

        return Y_m_i


class Updating_A_Layer(nn.Module):
    def __init__(self, epsilon):
        super(Updating_A_Layer,self).__init__()
        self.epsilon = epsilon

    def forward(self,Omega_array,R,thres):

        S,thres = outliers(self.epsilon,Omega_array,R,thres)

        return S, thres


class Updating_X_Layer(nn.Module):
    def __init__(self, Data_Size):
        super(Updating_X_Layer,self).__init__()
        self.Data_Size = Data_Size

    def forward(self,X,Omega,Gtr):

        G_cat = Ui2U(Gtr).reshape(self.Data_Size)
        Omega_C = 1 - Omega
        M_Omega = X * Omega
        X_rec = G_cat * Omega_C + M_Omega * Omega

        return X_rec