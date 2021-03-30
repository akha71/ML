%% exercise 1 solution complete
%% ======================= Part 2: Plotting =======================
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y);
plot (X, y, 'or')
hold on
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;
%%
%j1 = 0
%for i = 1:m;
%j1 = j1 + (X(i,:)*theta-y(i)).^2;
%J = (1/(2*m))*j1;
%end
%% vectorized way
prediction = X*theta
sqrerr = (prediction - y).^2
J = (1/(2*m))*sum(sqrerr)

%% =================== Part 3: Cost and Gradient descent ===================
num_iters = 1500;
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

err = prediction - y
theta0 = theta(1) - alpha/m*sum(err.*X(:,1))
theta1 = theta(2) - alpha/m*sum(err.*X(:,2))
theta = [theta0; theta1]



    % ============================================================

    % Save the cost J in every iteration  

   J_history(iter) = J;
prediction = X*theta
sqrerr = (prediction - y).^2
J = (1/(2*m))*sum(sqrerr)
   
   
end
plot(X(:,2), X*theta, '-')
iii = [1:1:1500]
figure;
plot (iii, J_history)