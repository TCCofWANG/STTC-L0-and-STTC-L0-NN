clear; close all; warning off;
addpath TensorRing
addpath tensor_toolbox
addpath tensorlab
addpath traffic_data  

% input data
Data = cell2mat(struct2cell(load("traffic_data\PeMS08\PeMs08.mat")));
adj = cell2mat(struct2cell(load("traffic_data\PeMS08\L_PeMS08.mat")));
Data_Size=size(Data);

% missing data 
MissingRatio = 0.3;
Omega=ones(Data_Size);
obs_idx=Omega(Omega==1);
obs_idx(randsample(Data_Size(1)*Data_Size(2)*Data_Size(3), floor(MissingRatio*Data_Size(1)*Data_Size(2)*Data_Size(3)))) = 0;
Omega(Omega==1)=obs_idx;
Data_Omega=Data.*Omega;

% model
model='STTC_CF';

% traffic data imputation
tic;
switch model
    case 'STTC_CF'
        [X,A] = STTC_CF(Data_Omega, Omega, adj);
    case 'STTC_L0'
        [X,A] = STTC_L0(Data_Omega, Omega, adj);            
end   
toc;
    
    
% evaluation
Omega_c=1-Omega;
[nmae,rmse,mae] = metrics(Data,X,Omega_c);

fprintf('\n model:%s, loss rate:%f, nmae:%f, mae:%f, rmse:%f, time:%f\n',model,MissingRatio,nmae,mae,rmse,toc);
