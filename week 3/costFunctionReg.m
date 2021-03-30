function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%first way
z = X*theta;
theta_temp = [0; theta(2:end, 1)]; % as no need to regularize x0
J = 1/m*(sum(-y.*log(sigmoid(z))-(1-y).*log(1 - sigmoid(z))) + 1/2*lambda*sum(theta_temp.^2)); %regularized cost function
error = sigmoid(z) - y;
grad = 1/m*((X'*error)+ lambda*theta_temp); %regularized gradient


%second way
%{
HY = zeros(size(theta));
z = X*theta;
HY = sigmoid(z);
AA = -y.*log(sigmoid(z))-(1-y).*log(1 - sigmoid(z));
% we need to separate j = 0 in regularization codes. as theta_0 wont be penalized. so we have: 
theta_temp = theta([2:size(theta)], 1);
BB = theta_temp.^2;
J = 1/m*(sum(AA)+ 1/2*lambda*sum(BB));
% we need to separate j = 0 in regularization codes. as theta_0 wont be penalized, so vector of grad must be written in 2 lines
error = HY - y;
grad (1,1) = 1/m*sum(error);
X_temp = X(:,[2:size(theta)]);
grad ([2:size(theta)],1) = 1/m*((X_temp'*error)+ lambda*theta_temp);
%}
% =============================================================

end
