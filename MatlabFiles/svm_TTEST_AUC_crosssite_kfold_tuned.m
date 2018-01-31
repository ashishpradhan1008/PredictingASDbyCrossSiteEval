% For stanford site, using pls_30 scores
%Training set --combined_rm_stanford_pls_30mat.csv  -----  
% Testing Set --- stanford_pls_30_mat.csv

%%% Similar to other sites.

rng(1,'twister') % for reproducibility
%
%  A= load('C:\ABIDE\preprocessed\pls_30\combined_pls_30mat.csv');% 171 control, 318 total
%   A1= A(:,1:3486);
%   groupsall= A(:,3487);
%   k=10;
% %  
 % stanford %
%   A= load('C:\ABIDE\preprocessed\pls_30\combined_rm_stanford_pls_30mat.csv');% 171 control, 318 total
%   A1= A(:,1:3486);
%   groupsall= A(:,3487);
%   k=10;
% % 
%  E= load('C:\ABIDE\preprocessed\pls_30\stanford_pls_30_mat.csv');% 18 control, 38 total
%   E1= E(:,1:3486);
%   groupsallE= E(:,3487);  % hc = 0,mdd = 1

% % sdsu %
%  A= load('C:\ABIDE\preprocessed\pls_30\combined_rm_sdsu_pls_30mat.csv');% 171 control, 318 total
%   A1= A(:,1:3486);
%   groupsall= A(:,3487);
%   k=10;
  

% % 
%  E= load('C:\ABIDE\preprocessed\pls_30\sdsu_pls_30_mat.csv');% 18 control, 38 total
%   E1= E(:,1:3486);
%   groupsallE= E(:,3487);  % hc = 0,mdd = 1%

% pitt %
%  A= load('C:\ABIDE\preprocessed\combined_rm_pitt_pls_30mat.csv');% 171 control, 318 total
%   A1= A(:,1:3486);
%   groupsall= A(:,3487);
%   k=10;
% % 
%   E= load('C:\ABIDE\preprocessed\pitt_pls_30_mat.csv');% 18 control, 38 total
%   E1= E(:,1:3486);
%   groupsallE= E(:,3487);  % hc = 0,mdd = 1
% 
% % 
% % kki %
%  A= load('C:\ABIDE\preprocessed\combined_rm_kki_pls_30mat.csv');% 171 control, 318 total
%   A1= A(:,1:3486);
%   groupsall= A(:,3487);
%   k=10;
% % 
%   E= load('C:\ABIDE\preprocessed\kki_pls_30_mat.csv');% 18 control, 38 total
%   E1= E(:,1:3486);
%   groupsallE= E(:,3487);  % hc = 0,mdd = 1

% % nyu (YOUNG) %

%  A= load('C:\ABIDE\preprocessed\pls_30\combined_rm_nyu_pls_30mat.csv');% 171 control, 318 total
%   A1= A(:,1:3486);
%   groupsall= A(:,3487);
%   k=10;
% % 
%   E= load('C:\ABIDE\preprocessed\pls_30\nyu_pls_30_mat.csv');% 18 control, 38 total
%   E1= E(:,1:3486);
%   groupsallE= E(:,3487);  % hc = 0,mdd = 1


% % olin %
%  A= load('C:\ABIDE\preprocessed\combined_rm_olin_pls_30mat.csv');% 171 control, 318 total
%   A1= A(:,1:3486);
%   groupsall= A(:,3487);
%   k=10;
% % 
%   E= load('C:\ABIDE\preprocessed\olin_pls_30_mat.csv');% 18 control, 38 total
%   E1= E(:,1:3486);
%   groupsallE= E(:,3487);  % hc = 0,mdd = 1
% 



indexall = zeros(1,size(A1,2));
count=0;

for i = 1:size(A1,2)
   indexall(1,i) = (count+1);
   count=count+1;
end;


% LEAVE ONE-OUT CROSSVALIDATION 
countFC = zeros(10,3486);



CVO = cvpartition(groupsall,'k',k);
cvFolds = crossvalind('Kfold', groupsall, k);

AUC = zeros(10,5);
  
 for j= 1:10

% /*# get indices of 10-fold CV

total_score_e= []; %zeros(k,4);

 for i = 1:k
    
   constraint=0;
   
     trainIdxnew = CVO.training(i);
    testIdxnew = CVO.test(i);        %# get indices training instances
    
    Atest = cat(2,A1,groupsall);
    
    Atest(testIdxnew, :) = [];
    
    groupsalltrain = Atest(:,3487);
     Anew = Atest(:,1:3486);
     
     
      % RANK THE VARIABLES BASED ON TTEST... 
     
      I = rankfeatures(Anew',groupsalltrain, 'Criterion', 'ttest',  'NumberOfIndices', 3486); %'CCWEIGHTING',0 );k
    %disp('done ranking');
    %[IXall,weights] = relieff(PBDHCnew,groupsalltest,5);
    
    newI = I';
    cnt =1;
    
    
     for l= 1:j*50
         countFC(j,I(l)) = countFC(j,I(l))+1;
      end;
    
   disp(newI(1));
    disp(newI(2));
   disp(newI(3));
  disp(newI(4));
  disp(newI(5));
%     
% SELECT THE TOP THREE VARIABLES 
    newIXAll = newI((j*50):size(A1,2));
     

   newA= cat(1,indexall,A1);
   
   newA1 = newA(:,newIXAll);
   
   newA1(1,:) = [];
%    
%    % another 10-fold for best c parameters 
%    c = cvpartition(length(groupsall(trainIdxnew)),'kfold',10);
   
   C = [  1, 10, 100];
   accscores = zeros(numel(C));
   
   for c = 1:numel(C)  
       cp = cvpartition(groupsall(trainIdxnew),'k',10); % Stratified cross-validation

       
       vals = crossval(@(XTRAIN, YTRAIN, XVAL, YVAL)(fun(XTRAIN, YTRAIN, XVAL, YVAL, C(c))),newA1(trainIdxnew,:),groupsall(trainIdxnew),'partition',cp); 
       accscores(c) = mean(vals);
   end

%// Then establish the C and S that gave you the bet f-score. Don't forget that c and s are just indexes though!
[cbest] = find(accscores == max(accscores(:)));
C_final = C(cbest);
if length(C_final)>1 
    constraint = C_final(1);
else
    constraint = C_final;
end;
   
 constraint


       %# train an SVM model over training instances
    svmModel = fitcsvm(newA1(trainIdxnew,:), groupsall(trainIdxnew), ...
                 'Standardize',true,'KernelFunction','linear','BoxConstraint',1 ,'KernelOffset',0);
             
    
    [label,score] = predict( svmModel,newA1(testIdxnew,:));  
    
    newmatrix = zeros(size(label,1),4);
    
    newmatrix(:,1) = double(groupsall(testIdxnew,:));
    newmatrix(:,2) = double(label);
    newmatrix(:,3)= score(:,1);
    newmatrix(:,4)= score(:,2);
    
     total_score_e = vertcat(total_score_e, newmatrix);
    
    D =[ groupsall(testIdxnew,:),label,score(:,1),score(:,2)];
    disp(D);
     
    
end;

  
[x1_e,y1_e,~,auc1] = perfcurve(total_score_e(:,1),total_score_e(:,4),1);
plot(x1_e,y1_e)

C = confusionmat(total_score_e(:,1),total_score_e(:,2));
CONF = [C(2,2) C(1,2) C(1,1) C(2,1)];
AUC(j,1) = auc1;
AUC(j,2) = C(2,2);
AUC(j,3) = C(1,2);
AUC(j,4) = C(1,1);
AUC(j,5) = C(2,1);
  disp('done!');
end;

%% 
% Using consensus features, first find the features and then apply to the
% test data set
% To run the code uncomment it ...

%  I = find(countFC(2,:) > 7);
%     % % 
%  reducedA =  A1(:,I);
%     
%  C = [  1, 10, 100];
%  accscores = zeros(numel(C));
%        
%  for c = 1:numel(C)  
%     cp = cvpartition(groupsall,'k',10); % Stratified cross-validation
%     
%            vals = crossval(@(XTRAIN, YTRAIN, XVAL, YVAL)(fun(XTRAIN, YTRAIN, XVAL, YVAL, C(c))),reducedA,groupsall,'partition',cp); 
%            accscores(c) = mean(vals);
%        end
%     
%     %// Then establish the C and S that gave you the best accuracy
%     
%     [cbest] = find(accscores == max(accscores(:)));
%     C_final = C(cbest);
%     if length(C_final)>1 
%         constraint = C_final(1);
%     else
%         constraint = C_final;
%     end;
%        
%                        
%     
%     svmFinalModel = fitcsvm(reducedA, groupsall, 'Standardize',true,'KernelFunction','linear','BoxConstraint',constraint,'KernelOffset',0);
%       [labelE,scoreE] = predict( svmFinalModel,E1(:,I)); 
%       C1 = confusionmat(double(groupsallE),double(labelE));
%      [x1_e,y1_e,~,auc1] = perfcurve(double(groupsallE),scoreE(:,2),1);
%     
