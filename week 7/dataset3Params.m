function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_choice = [0.01 0.03 0.1 0.3 1 3 10 30]';
sig_choice = [0.01 0.03 0.1 0.3 1 3 10 30]';
error_min = Inf;
predictions = zeros(length(yval),1);

for i = 1:length(C_choice)
  for j = 1:length(sig_choice)
    
    model= svmTrain(X, y, C_choice(i), @(x1, x2) gaussianKernel(x1, x2, sig_choice(j))); %training using x and y in training set, for all combinations of C and sigma
    predictions = svmPredict(model, Xval); %returns the prediction using Xval
    error = mean(double(predictions ~= yval)); %comparing our predictions (in validation set) with yval (real labels in validation set)
    
    if error < error_min  %updating minimum error 
        error_min = error;
        C = C_choice(i); %updating the best C
        sigma = sig_choice(j); %updating the best sigma
    endif 

  endfor
endfor



% =========================================================================

end
