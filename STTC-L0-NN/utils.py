import torch
import tensorly as tl
from tensorly.base import unfold
import scipy.io
from scipy.linalg import toeplitz
tl.set_backend('pytorch')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def def_para():
    para = {}
    X_data = scipy.io.loadmat('data/PeMS08/PeMs08.mat')
    variable_names = [key for key in X_data.keys() if not key.startswith('__')]
    X = torch.tensor(X_data[variable_names[0]],dtype=torch.float32)
    Data_Size = X.size()
    adj_data = scipy.io.loadmat('data/PeMS08/L_PeMS08.mat')
    variable_names = [key for key in adj_data.keys() if not key.startswith('__')]
    adj = torch.tensor(adj_data[variable_names[0]],dtype=torch.float32)
    initial_seed = 2
    torch.manual_seed(initial_seed)


    para['training set size'] = 10
    para['testing set size'] = 1
    para['MissingRatio'] = 0.5
    para['Data_Size'] = X.size()
    para['r'] = [15, 15, 15]
    para['epsilon'] = 1e-50

    para['alpha'] = [1e-8, 1e-8, 1e-8]
    para['lambda'] = [0.1, 0.1, 0.1]
    para['mu'] = [1e+0,1e+3,1e+4]
    para['ni'] = [1e-2,1e-2,1e-2]

    para['tol'] = 1e-6
    para['inner_tol'] = 1e-3
    para['Block_num'] = 5
    para['lr'] = 1e-6
    scale = 10000

    dim = len(Data_Size)
    para['dim'] = dim
    I = X.size()

    Gtr = [None] * len(Data_Size)
    G_m = [None] * len(Data_Size)
    R_m = [None] * len(Data_Size)
    R_m_size = [None] * len(Data_Size)

    for i in range(dim - 1):
        Gtr[i] = torch.randn(para['r'][i], Data_Size[i], para['r'][i + 1])
        G_m[i] = Gtr[i].reshape(Data_Size[i], -1)
    Gtr[dim - 1] = torch.randn(para['r'][dim - 1], Data_Size[dim - 1], para['r'][0])
    G_m[dim - 1] = Gtr[dim - 1].view(-1, para['r'][0] * para['r'][1])

    L = [None] * len(Data_Size)
    L[0] = adj
    for i in range(1,dim):
        ctoe = [-1, 1] + [0] * (Data_Size[i] - 2)
        rtoe = [-1] + [0] * (Data_Size[i] - 2)
        Toep = torch.tensor(toeplitz(ctoe, rtoe), dtype=torch.float32)
        L[i] = torch.mm(Toep, Toep.T)

    for i in range(dim):
        R_m[i] = torch.inverse(2 * para['lambda'][i] * L[i] + para['ni'][i] * torch.eye(I[i], dtype=torch.float32))
        R_m_size[i] = R_m[i].size()

    A = torch.zeros(Data_Size)

    train_data = {'Omega': [], 'Gtr': {}, 'G_cat': {}, 'X_rec': {}, 'G_m': {}, 'R_m': {},'A': []}
    test_data = {'Omega': [], 'Gtr': {}, 'G_cat': {}, 'X_rec': {}, 'G_m': {}, 'R_m': {},'A': []}

    print('Loading Data...')

    for i in range(para['training set size']):
        Omega = torch.ones(Data_Size)
        num_elements = torch.prod(torch.tensor(Data_Size)).item()
        num_zero_elements = int(para['MissingRatio'] * num_elements)
        random_indices = torch.randperm(num_elements)[:num_zero_elements]
        Omega.view(-1)[random_indices] = 0
        train_data['Omega'].append(Omega)
        torch.manual_seed(initial_seed + 1)
        train_data['Gtr'][f'Gtr{i}'], train_data['G_cat'][f'G_cat{i}'], train_data['X_rec'][f'X_rec{i}'] = precomputation(X,Gtr, dim, Data_Size, train_data['Omega'][i])
        train_data['G_m'][f'G_m{i}'] = G_m
        train_data['R_m'][f'R_m{i}'] = R_m
        train_data['A'].append(A)


    for i in range(para['testing set size']):
        Omega = torch.ones(Data_Size)
        num_elements = torch.prod(torch.tensor(Data_Size)).item()
        num_zero_elements = int(para['MissingRatio'] * num_elements)
        random_indices = torch.randperm(num_elements)[:num_zero_elements]
        Omega.view(-1)[random_indices] = 0
        test_data['Omega'].append(Omega)
        torch.manual_seed(initial_seed + 1)
        test_data['Gtr'][f'Gtr{i}'], test_data['G_cat'][f'G_cat{i}'], test_data['X_rec'][f'X_rec{i}'] = precomputation(X, Gtr, dim, Data_Size,test_data['Omega'][i])
        test_data['G_m'][f'G_m{i}'] = G_m
        test_data['R_m'][f'R_m{i}'] = R_m
        test_data['A'].append(A)

    sizes = [size[0] for size in R_m_size]
    para['W0'] = torch.ones(sizes[0], sizes[0])
    para['B0'] = torch.zeros(sizes[0], sizes[0])
    para['W1'] = torch.ones(sizes[1], sizes[1])
    para['B1'] = torch.zeros(sizes[1], sizes[1])
    para['W2'] = torch.ones(sizes[2], sizes[2])
    para['B2'] = torch.zeros(sizes[2], sizes[2])

    def move_to_device(data, device):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {key: move_to_device(value, device) for key, value in data.items()}
        elif isinstance(data, list):
            return [move_to_device(item, device) for item in data]
        else:
            return data

    para = move_to_device(para, device)
    X = move_to_device(X, device)
    train_data = move_to_device(train_data, device)
    test_data = move_to_device(test_data, device)
    print('Data loaded')

    return para,X,train_data,test_data,scale


def precomputation(X,Gtr,dim,Data_Size,Omega):

    Omega_C = 1 - Omega
    M_Omega = X * Omega
    for j in range(3):
        for i in range(dim):
            if i == 0:
                F = Gtr
            if i == 1:
                F = Gtr[1:] + Gtr[:1]
            if i == 2:
                F = Gtr[2:] + Gtr[:2]
            Gtr[i] = precompute(tenspermute(Omega, i), F, tenspermute(M_Omega, i))
    G_cat = Ui2U(Gtr).view(Data_Size)
    X_rec = G_cat * Omega_C + M_Omega * Omega

    return Gtr,G_cat,X_rec
def precompute(Omega, U, X_Omega):

    B = TCP(U[1:])
    alpha = 1e-8
    P = torch.tensor(unfold(tl.tensor(Omega).clone().detach().requires_grad_(True), 0))  # I1 * (I2...In)
    X = torch.tensor(unfold(tl.tensor(X_Omega).clone().detach().requires_grad_(True), 0))  # I1 * (I2...In)
    rn, I1, r1 = U[0].shape
    U1_temp = U[0]
    U1 = torch.zeros((rn, I1, r1))

    for i in range(I1):
        idx = torch.nonzero(P[i, :] == 1).squeeze()
        if len(idx) == 0:
            continue
        A = torch.tensor(unfold(tl.tensor(tenspermute(B[:, idx, :],1)).clone().detach().requires_grad_(True),0))
        y = X[i, idx].reshape(-1, 1)
        u1_temp = U1_temp[:, i, :].reshape(rn * r1, 1)
        left_matrix = A.T @ A + alpha * torch.eye(A.T.shape[0], dtype=torch.float32)
        right_vector = A.T @ y + alpha * u1_temp
        U1[:, i, :] = torch.linalg.solve(left_matrix, right_vector).reshape(rn, r1)

    return U1

def right_unfold(U):

    l, c, r = U.shape
    RU = U.reshape(l, c * r)

    return RU

def left_unfold(U):

    l, c, r = U.shape

    LU = U.reshape(l * c, r)

    return LU

def T2M(T):

    k1, k2 = T.size()
    M = T.view(k1, k2)

    return M

def TCP(U):

    n = len(U)
    T = U[0]
    if n == 1:
        return T
    for i in range(1, n):
        lT, cT, _ = T.shape
        _, cU, rU = U[i].shape
        T_unfolded = torch.mm(left_unfold(T), right_unfold(U[i]))
        T = T_unfolded.view(lT, cT * cU, rU)

    return T

def tens2mat(tensor, mode_row, mode_col):

    tensor_shape = tensor.shape
    row_shape = tensor_shape[mode_row]
    num_rows = torch.prod(torch.tensor(row_shape))
    T = tensor.reshape(num_rows, -1)

    return T

def mat2tens(matrix, size_tens, mode_row, mode_col):

    tensor = matrix.view(size_tens)

    return tensor

def tenspermute(T, i):

    if i == 0:
        TPi = T
    if i == 1:
        TPi = T.permute(1,2,0)
    if i == 2:
        TPi = T.permute(2,0,1)

    return TPi

def Ui2U(Ui):

    T = TCP(Ui)
    d = len(Ui)
    I = [Ui[i].size(1) for i in range(d)]
    T_reshaped = T.permute(1, 0, 2)
    v = torch.einsum('ijj->i', T_reshaped)
    U = v.reshape(I)

    return U

def inverse(A,B):

    return torch.linalg.solve(B.T, A.T).T

class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, output, target):

        return (torch.norm(target - output, 'fro') ** 2) / (torch.norm(target, 'fro') ** 2)

def outliers(epsilon,Omega_array,R,thres):

    R_Omega_nonzero = R[Omega_array != 0]
    n_abs = torch.abs(R_Omega_nonzero)
    n_sort, _ = torch.sort(n_abs)
    Noise = R_Omega_nonzero.sort()[0]
    A_E = R_Omega_nonzero.std()
    IQR = torch.quantile(Noise, 0.95)
    sigma = 1.06 * min(A_E, IQR / 1.34) * len(R_Omega_nonzero) ** -0.2
    w = torch.exp(-torch.abs(R_Omega_nonzero) / sigma)
    position = w < epsilon
    outliers = n_abs[position]
    if outliers.numel() > 0:
        temp_thres = thres.clone()
        thres = torch.min(torch.min(outliers) ** 2, temp_thres)
        R_Omega = R[Omega_array != 0].view(-1, 1)
        s = R_Omega * (torch.abs(R_Omega) >= torch.sqrt(thres))
        S = torch.zeros_like(R)
        S[Omega_array != 0] = s
    else:
        S = torch.zeros_like(R)

    return S,thres

def metrics(X, X_approx, idx_recover):

    X = X.to(torch.float)
    X_approx = X_approx.to(torch.float)
    idx_recover = idx_recover.to(torch.float)
    abs_diff = torch.abs((X - X_approx) * idx_recover)
    rmae = torch.sum(abs_diff) / torch.sum(X * idx_recover)
    squared_diff = ((X - X_approx) * idx_recover) ** 2
    mean_squared_diff = torch.sum(squared_diff) / torch.sum(idx_recover)
    rmse = torch.sqrt(mean_squared_diff)
    mae = torch.sum(abs_diff) / torch.sum(idx_recover)

    return rmae, rmse, mae
