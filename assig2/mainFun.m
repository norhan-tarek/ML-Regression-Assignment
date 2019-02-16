clc
Htrain= heartDD(1:200,:);
Hcv= heartDD(201:226,:);
Htest= heartDD(227:250,:);

%Training phase
Alpha=.01;
m=length(Htrain(:,1));
U=Htrain(:, 3:5);
U1=Htrain(:, 12:13);
U2=Htrain(:, 1:7);
U3=Htrain(:, 4:5);
U4=Htrain(:, 8);

XL=[ones(m,1) U2];
XL1=[ones(m,1) U2 U1 U2.^2 U2.^3 U2.^4];
XL2=[ones(m,1) U U1 U2.^2 U.^2 U.^3 U.^4 U.^5 U.^6 U.^7 U.^8 U.^9];
XL3=[ones(m,1) exp(-U2)];

[EL,thetaL]=LogisticRegFun( XL,Htrain,m,Alpha);
[EL1,thetaL1]=LogisticRegFun( XL1,Htrain,m,Alpha);
[EL2,thetaL2]=LogisticRegFun( XL2,Htrain,m,Alpha);
[EL3,thetaL3]=LogisticRegFun( XL3,Htrain,m,Alpha);

%cross validation phase
n=length(Hcv(:,1));
Ucv=Hcv(:, 3:5);
U1cv=Hcv(:, 12:13);
U2cv=Hcv(:, 1:7);
U3cv=Hcv(:, 4:5);
U4cv=Hcv(:, 8);

XLcv=[ones(n,1) U2cv];
XL1cv=[ones(n,1) U2cv U1cv U2cv.^2 U2cv.^3 U2cv.^4];
XL2cv=[ones(n,1) Ucv U1cv U2cv.^2 Ucv.^2 Ucv.^3 Ucv.^4 Ucv.^5 Ucv.^6 Ucv.^7 Ucv.^8 Ucv.^9];
XL3cv=[ones(n,1) exp(-U2cv)];
JLcv(1)= CostLogisticFun(XLcv,Hcv,thetaL,n);
JLcv(2)= CostLogisticFun(XL1cv,Hcv,thetaL1,n);
JLcv(3)= CostLogisticFun(XL2cv,Hcv,thetaL2,n);
JLcv(4)= CostLogisticFun(XL3cv,Hcv,thetaL3,n);

%Testing phase
i=length(Htest(:,1));
ULtest2=Htest(:, 1:7);
ULtest1=Htest(:, 12:13);

BestFun=find(JLcv==min(JLcv))
XLtest=[ones(i,1) ULtest2 ULtest1 ULtest2.^2 ULtest2.^3 ULtest2.^4];
Jtest=CostLogisticFun(XLtest,Htest,thetaL1,i)

%ploting the 4 hypothesis
figure (4)
plot(EL)
hold on
plot(EL1,'r')
plot(EL2,'g')
plot(EL3,'y')
hold off