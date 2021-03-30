
data = load ex2data1.txt

Function g = sigmoid (z)
g = zeros(size(z));
g = 1./(1+e.^(-z));
end