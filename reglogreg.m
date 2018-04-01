% printing option
more off;

% read files
D_tr = csvread('spambasetrain.csv'); 
D_ts = csvread('spambasetest.csv');  

% construct x and y for training and testing
X_tr = D_tr(:, 1:end-1);
y_tr = D_tr(:, end);
X_ts = D_ts(:, 1:end-1);
y_ts = D_ts(:, end);

% number of training / testing samples
n_tr = size(D_tr, 1);
n_ts = size(D_ts, 1);

% add 1 as a feature
X_tr = [ones(n_tr, 1) X_tr];
X_ts = [ones(n_ts, 1) X_ts];

% perform gradient descent :: logistic regression
n_vars = size(X_tr, 2);              % number of variables
lr = 1e-3;                           % learning rate
w = zeros(n_vars, 1);                % initialize parameter w
w_new = zeros(n_vars, 1);   
tolerance = 1e-2;                    % tolerance for stopping criteria

iter = 0;                            % iteration counter
max_iter = 1000;                     % maximum iteration

#Without Regularization 

while true
  iter = iter + 1;                 % start iteration
  a=X_tr*w;
  grad = zeros(n_vars, 1);         % initialize gradient
  % calculate gradient
  P=1./(1+exp(-a));
        % grad(j) = ....             % compute the gradient with respect to w_j here
  grad=X_tr'*(y_tr-P);
        % take step
        % w_new = w + .....              % take a step using the learning rate
  w_new=w+lr*grad;
  fflush(stdout);
  printf('iter = %d, mean abs gradient = %0.3f\n', iter, mean(abs(grad)));

      % stopping criteria and perform update if not stopping
  if mean(abs(grad)) < tolerance
    w = w_new;
    break;
  else
    w = w_new;
  end
  if iter >= max_iter 
     break;
  end
end
      
  % use w for prediction
pred = zeros(n_ts, 1);               % initialize prediction vector
b_spam = zeros(n_ts, 1); 
b_spam=X_ts*w;
b_spam=1./(1+exp(-b_spam));

  % pred(i) = .....                % compute your prediction
pred=(b_spam>=0.5);
  
   % calculate testing accuracy
   % ...
test_error=(mean(pred~=y_ts))*100;
test_accuracy=100-test_error;
printf('test_accuracy = %d \n', test_accuracy);


% repeat the similar prediction procedure to get training accuracy
   % ...

% use w for prediction
pred_train = zeros(n_tr, 1);               % initialize prediction vector
b_spam_train = zeros(n_tr, 1); 
b_spam_train=X_tr*w;
b_spam_train=1./(1+exp(-b_spam_train));
     % pred(i) = .....                % compute your prediction
pred_train=(b_spam_train>=0.5);


% calculate training accuracy
% ...
train_error=(mean(pred_train~=y_tr))*100;
train_accuracy=100-train_error;
printf('train_accuracy = %d \n', train_accuracy);


#Regularization   
k=[-8,-6,-4,-2,0,2];
train_accuracy_vector=[];
test_accuracy_vector=[];
iter = 0;                            % iteration counter
for each_k = k
  w = zeros(n_vars, 1);                % initialize parameter w
  w_new = zeros(n_vars, 1);
  iter = 0;                            % iteration counter
  while true
    lambda=power(2,each_k);
    iter = iter + 1;                 % start iteration
    a = zeros(n_tr, 1); 
    a=X_tr*w;
    grad = zeros(n_vars, 1);         % initialize gradient
      % calculate gradient
    P=1./(1+exp(-a));
        % grad(j) = ....             % compute the gradient with respect to w_j here
    grad=X_tr'*(y_tr-P);
    % take step
    % w_new = w + .....              % take a step using the learning rate
    w_new=w+lr*(grad-lambda*w);
    fflush(stdout);
    printf('iter = %d, mean abs gradient = %0.3f\n', iter, mean(abs(grad)));

    % stopping criteria and perform update if not stopping
    if mean(abs(grad)) < tolerance
      w = w_new;
      break;
    else
      w = w_new;
    end
    if iter >= max_iter 
      break;
    end
  end
      
  % use w for prediction
  pred_r = zeros(n_ts, 1);               % initialize prediction vector
  b_spam_r = zeros(n_ts, 1); 
  b_spam_r=X_ts*w;
  b_spam_r=1./(1+exp(-b_spam_r));
  % pred(i) = .....                % compute your prediction
  pred_r=(b_spam_r>=0.5);
      
   % calculate testing accuracy
   % ...
  test_error_r=(mean(pred_r~=y_ts))*100;
  test_accuracy_r=100-test_error_r;
  test_accuracy_vector=[test_accuracy_vector,test_accuracy_r];

   % repeat the similar prediction procedure to get training accuracy
   % ...

   % use w for prediction
  pred_train_r = zeros(n_tr, 1);               % initialize prediction vector
  b_spam_train_r = zeros(n_tr, 1); 
  b_spam_train_r=X_tr*w;
  b_spam_train_r=1./(1+exp(-b_spam_train_r));
  % pred(i) = .....                % compute your prediction

  pred_train_r=(b_spam_train_r>=0.5);

   % calculate training accuracy
   % ...
  train_error_r=(mean(pred_train_r~=y_tr))*100;
  train_accuracy_r=100-train_error_r;
  train_accuracy_vector=[train_accuracy_vector,train_accuracy_r];
end

k=[-8,-6,-4,-2,0,2];
plot(k,test_accuracy_vector,'b');
hold on
plot(k,train_accuracy_vector,'r');
xlabel('k');
ylabel('Accuracy');
title("Accuracy plot");
legend('Test Accuracy','Train Accuracy');
