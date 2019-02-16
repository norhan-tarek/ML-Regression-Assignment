function [ E,Theta ] = LogisticRegFun( X,T,m,Alpha )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

n=length(X(1,:));
Theta=zeros(n,1);
k=1;

[E(k),X,Y]=CostLogisticFun( X,T,Theta,m);

R=1;
while R==1
Alpha=Alpha*1;
Theta=Theta-(Alpha/m)*X'*(X*Theta-Y);
k=k+1;
h = sigmoid(X*Theta);
E(k)=-(1/m) * sum( Y .* log(h) + (1-Y) .* log(1-h) );
if E(k-1)-E(k)<0
    break
end 
q=(E(k-1)-E(k))./E(k-1);
if q <.0001;
    R=0;
end
end

end

