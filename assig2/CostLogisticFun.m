function [ J,x,y ] = CostLogisticFun( x,T,Theta,m )
 n=length(x(1,:));
for w=2:n
    if max(abs(x(:,w)))~=0
    x(:,w)=(x(:,w)-mean((x(:,w))))./std(x(:,w));
    end
end

y=T(:,14);

h = sigmoid(x*Theta);
J = -(1/m) * sum( y .* log(h) + (1-y) .* log(1-h) );


end

