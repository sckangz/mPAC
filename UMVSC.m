function [res,FV,obj,Wv]=UMVSC(X,s,B,alpha,beta,gamma,FV,SV,LV)
% s is the true class label.
opts.record = 0;
opts.mxitr  = 1000;%1000
opts.xtol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;
out.tau = 1e-3;

v1=length(X);
% FV=cell(v1,v2);
% 
% SV=cell(v1,v2);
Wv=ones(v1,1)/v1;
% LV=cell(v1,v2);
RV=cell(v1,1);
n=size(X{1},1);  % # of samples
c=length(unique(s));% # of clusters
 
%     Fv=orth(rand(n,c));
Rv=eye(c);
%     Sv=eye(n);
sumFv=zeros(n,c);
for num = 1:v1
%   FV{num}=Fv;
%   SV{num}=Sv;
    RV{num}=Rv;
    sumFv=sumFv+FV{num}*RV{num}/Wv(num);
end
% Y=Fv*Rv>0;
Y=zeros(n,c);
[~, l] = max(sumFv, [], 2);
Y = full(sparse(1:n, l, ones(n, 1), n, c));

for i=1:100
    lold=l;
    for num = 1:v1
%         Zv=ZV{num};
        xv=X{num};
        xv=xv';
        Fv=FV{num};
        Rv=RV{num};
        xx=xv'*xv;
        parfor ij=1:n
                d=distance(Fv,n,ij);
                Sv(:,ij)=B{num}*(xx(:,ij) - beta/4*d'); 
        end
%         Zv(find(Zv<0))=0;
%         Zv=(Zv+Zv')/2;
        Sv=Sv-diag(diag(Sv));
        D = diag(sum(Sv));
        Lv = D-Sv;
        [Fv,out]= solveF(Fv,@fun1,opts,gamma/Wv(num)/beta,Y,Rv,Lv);
       
        [a b d]=svd(Fv'*Y);
        Rv=a*d';
        Q(num)=norm(Y-Fv*Rv,'fro');
        FV{num}=Fv;
        SV{num}=Sv;
        LV{num}=Lv;        
    end
    for j=1:v1
        Wv(j)= Q(j)/sum(Q);
        Fv=FV{j};
        Rv=RV{j};
        PV(:,:,j)=Fv*Rv/Wv(j);
    end
   
    P=sum(PV,3);
    Y=zeros(n,c);
   
%     for ji=1:n
%         [v,j]=max(P(ji,:));
%         Y(ji,j)=1;
%         l(ji)=j;
%     end
    
    [~, l] = max(P, [], 2);
    Y = full(sparse(1:n, l, ones(n, 1), n, c));
    
    term=0;
    for num=1:v1
        term=term+(norm(X{num}'-X{num}'*SV{num},'fro'))^2+alpha*(norm(SV{num},'fro'))^2 + beta*trace(FV{num}'*LV{num}*FV{num}) + gamma/Wv(num)*(norm(Y-FV{num}*RV{num}','fro'))^2;
    end
    obj(i)=term;      
    if i>4&&(norm(l-lold)/norm(l)<1e-5)
        break
    end
end
[res] = Clustering8Measure(l,s);
end

function [all]=distance(F,n,ij)
    for ji=1:n
        all(ji)=(norm(F(ij,:)-F(ji,:)))^2;
    end
end

function [F,G]=fun1(P,alpha,Y,Q,L)
    G=2*L*P-2*alpha*Y*Q';
    F=trace(P'*L*P)+alpha*(norm(Y-P*Q,'fro'))^2;
end