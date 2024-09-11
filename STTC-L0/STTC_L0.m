function [X_rec,A] = STTC_L0(M_Omega, Omega, adj)

r=[15,15,15];
Data_Size = size(M_Omega);
I=Data_Size;
max_iter=500;
inner_iter=20;
tol=1e-6;
inner_tol=1e-3;
lambda=[0.1,0.1,0.1];
alpha=[1e-8,1e-8,1e-8];
rho=1.5;
dim = length(Data_Size);
scale = 10000;
epsilon=1e-50;
Omega_c=1-Omega;


% Initialization
Gtr=cell(size(Data_Size,2),1);
G_m=cell(size(Data_Size,2),1);%matrix: Gn(2)
Q_m=cell(size(Data_Size,2),1);
U_m=cell(size(Data_Size,2),1);
Z_m=cell(size(Data_Size,2),1);
V_m=cell(size(Data_Size,2),1);


for i=1:dim-1
    Gtr{i}=rand(r(i),Data_Size(i),r(i+1));
    G_m{i}=double(tenmat(Gtr{i},2));

end
Gtr{dim}=rand(r(dim),Data_Size(dim),r(1));
G_m{dim}=double(tenmat(Gtr{dim},2));


% feature matrix
L=cell(size(Data_Size,2),1);
L{1}=adj;
for i=2:dim
    ctoe=[-1,1,zeros(1,Data_Size(i)-2)];
    rtoe=[-1,zeros(1,Data_Size(i)-2)];
    Toep=toeplitz(ctoe,rtoe);%Toeplitz(0, 1, -1)
    L{i}=Toep*Toep';
end

%precompute
A=zeros(Data_Size);
for j = 1 : 3
    for i=1:dim
        Gtr{i} = precompute(TensPermute(Omega, i), Gtr([i:dim, 1:i-1]),TensPermute(M_Omega, i));
    end
end

% traffic data imputation
iter =1;
G_cat=reshape(Ui2U(Gtr), Data_Size);
X_rec=G_cat.*Omega_c+(M_Omega).*Omega;
while iter < max_iter

    G_pre=G_m;X_pre=X_rec;
    N=X_rec-A;
    for i=1:dim
        GtrT=Gtr([i:dim, 1:i-1]);
        [r_n, In, r_n1] = size(GtrT{1});
        C = tens2mat(TensPermute(TCP(GtrT(2:end)),2),1,2:dim); 
        N_m=tens2mat(TensPermute(N,i),1,2:dim);
        Z_m{i}=G_m{i};
        V_m{i}=Z_m{i}*0;
        Q_m{i}=G_m{i};
        U_m{i}=Q_m{i}*0;
        ni=[1e-2,1e-2,1e-2];% tuned
        mu=[1e+0,1e+3,1e+4];% tuned

        for j=1:inner_iter

            Gmi_pre=G_m{i};

            G_m{i}=(N_m*C+0.5*mu(i)*Z_m{i}-0.5*V_m{i}+0.5*ni(i)*Q_m{i}-0.5*U_m{i}+alpha(i)*G_pre{i})/(C'*C+(alpha(i)+0.5*mu(i)+0.5*ni(i))*eye(r_n*r_n1)); 
            
            Q_m{i}=(2*lambda(i)*L{i}+ni(i)*eye(I(i)))\(ni(i)*G_m{i}+U_m{i});
    
            Z=G_m{i}+V_m{i}/mu(i);
            Z(Z<0)=0;
            Z_m{i}=Z;

            U_m{i}=U_m{i}+ni(i)*(G_m{i}-Q_m{i});

            V_m{i}=V_m{i}+mu(i)*(G_m{i}-Z_m{i});

            ni(i)=min([1.5*ni(i),1e+8]);

            mu(i)=min([1.5*mu(i),1e+8]);

            d_GG=sum((G_m{i}-Gmi_pre).^2,"all")/sum((Gmi_pre).^2,"all");
            d_QG=sum((G_m{i}-Q_m{i}).^2,"all")/sum((G_m{i}).^2,"all");
            d_ZG=sqrt(sum((G_m{i}-Z_m{i}).^2,"all"))/sum((G_m{i}).^2,"all");
            s_gg=[d_GG,d_QG,d_ZG];

            if max(s_gg)<inner_tol
                break;
            end


        end

    Gtr{i}=permute(mat2tens(G_m{i},[In, r_n,r_n1],1,[2,3]),[2,1,3]);
    
    end

    R = X_rec - G_cat ;
    [A,scale] = solve_l0(R,Omega,scale,epsilon);

    G_cat=reshape(Ui2U(Gtr), Data_Size);
    X_rec=G_cat.*Omega_c+M_Omega.*Omega;    

    d_X=sum((X_rec-X_pre).^2,"all")/sum((X_pre).^2,'all');

    if d_X < tol
        break
    end

    iter =iter+1;
    
end

end


function [S,thres]=solve_l0(R,Omega_array,thres,epsilon)
    Omega = find(Omega_array==1);
    posi = Laplace_kernel(abs(R(Omega)),epsilon);
    outliers_posi = Omega(posi);
    outliers = abs(R(outliers_posi));
    if ~isempty(outliers)
        temp_thres = thres;
        thres = min(min(outliers(:))^2,temp_thres);
        s = reshape(R(Omega),[],1).* (reshape(abs(R(Omega)),[],1) >= sqrt(thres));
        S = zeros(size(R));
        S(Omega) = s;
    else
        S=zeros(size(R));
    end

end

function position = Laplace_kernel(noise,epsilon)
    Noise = noise;
    Noise = sort(Noise);
    A_E = std(noise);
    R = prctile(Noise,95);
    sigma = 1.06*min(A_E,R/1.34)*length(noise)^-0.2;
    w = exp(-(abs(noise)./(sigma)));
    position = find(w<epsilon);  
end