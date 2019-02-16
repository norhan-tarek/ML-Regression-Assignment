clc

%training phase
T=housedatacomplete(1:12969,:);
Tcv=housedatacomplete(12970:17292,:);
Ttest=housedatacomplete(17293:21613,:);
Alpha=.01;

m=length(T(:,1));
U0=T(:,2);
U=double(T(:,4:19));
U1=double(T(:,20:21));

X=[ones(m,1) U U1];
X1=[ones(m,1) U U1 U.^2 U.^3];
X2=[ones(m,1) U U1  U.^2 U.^3 U.^4 U.^5 U.^6 ];
X3=[ones(m,1) U U1  U.^2 U.^3 U.^4 U.^5 U.^6 U.^7 U.^8 U.^9 U.^10];
poly=[1,3,6,10];

[E,theta]=RegressionFun( X,T,m,Alpha);
[E1,theta1]=RegressionFun( X1,T,m,Alpha );
[E2,theta2]=RegressionFun( X2,T,m,Alpha);
[E3,theta3]=RegressionFun( X3,T,m,Alpha);

%ploting the 4 hypothesis 
figure (2)
plot(E)
hold on
plot(E1,'r')
plot(E2,'g')
plot(E3,'y')
hold off
J=[min(E),min(E1),min(E2),min(E3)];

%cross validation phase
Ucv=double(Tcv(:,4:19));
U1cv=double(Tcv(:,20:21));
n=length(Tcv(:,1));
Xcv=[ones(n,1) Ucv U1cv];
X1cv=[ones(n,1) Ucv U1cv Ucv.^2 Ucv.^3];
X2cv=[ones(n,1) Ucv U1cv  Ucv.^2 Ucv.^3 Ucv.^4 Ucv.^5 Ucv.^6 ];
X3cv=[ones(n,1) Ucv U1cv  Ucv.^2 Ucv.^3 Ucv.^4 Ucv.^5 Ucv.^6 Ucv.^7 Ucv.^8 Ucv.^9 Ucv.^10];
Ycv=double(Tcv(:,3))/mean(double(Tcv(:,3)));
Jcv= zeros(1,4);
Jcv(1) = CostFunLinear(Xcv,Tcv,theta,n);
Jcv(2) = CostFunLinear(X1cv,Tcv,theta1,n);
Jcv(3)= CostFunLinear(X2cv,Tcv,theta2,n);
Jcv(4) = CostFunLinear(X3cv,Tcv,theta3,n);
y=double(Tcv(:,3));
thetaN= inv(transpose(X2cv)*X2cv)*(transpose(X2cv)*y);
JNorm=CostFunLinear(X2cv,Tcv,thetaN,n)
figure (3)
plot (poly,J)
hold on
plot(poly,Jcv,'r')
legend('training','cross validation');
hold off
%Testing phase
Utest=double(Ttest(:,4:19));
U1test=double(Ttest(:,20:21));
i=length(Ttest(:,1));
Xtest=[ones(i,1) Utest U1test  Utest.^2 Utest.^3 Utest.^4 Utest.^5 Utest.^6 ];
Jtest = CostFunLinear(Xtest,Ttest,theta2,i)
