function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y);
J = 0;
grad = zeros(size(theta));


HY = zeros(size(theta));
HY = X*theta;
AA = -y.*log(sigmoid(HY))-(1-y).*log(1 - sigmoid(HY));
theta_temp = theta([2:size(theta)], 1);
BB = theta_temp.^2;
J = 1/m*(sum(AA)+ 1/2*lambda*sum(BB));

error = sigmoid(HY) - y;
grad (1,1) = 1/m*sum(error);
grad(2:size(theta),1) = 1/m*((X'*error)+ lambda.*theta_temp);

end