 
rng(1,'twister') % for reproducibility

    I = find(countFC(7,:) > 9);
    % 
     reducedA =  A1(:,I);
    
    C = [  1, 10, 100];
       accscores = zeros(numel(C));
       
       for c = 1:numel(C)  
           cp = cvpartition(groupsall,'k',10); % Stratified cross-validation
    
           
           vals = crossval(@(XTRAIN, YTRAIN, XVAL, YVAL)(fun(XTRAIN, YTRAIN, XVAL, YVAL, C(c))),reducedA,groupsall,'partition',cp); 
           accscores(c) = mean(vals);
       end
    
    %// Then establish the C and S that gave you the best accuracy
    
    [cbest] = find(accscores == max(accscores(:)));
    C_final = C(cbest);
    if length(C_final)>1 
        constraint = C_final(1);
    else
        constraint = C_final;
    end;
       
                       
    
    svmFinalModel = fitcsvm(reducedA, groupsall, 'Standardize',true,'KernelFunction','linear','BoxConstraint',constraint,'KernelOffset',0);
      [labelE,scoreE] = predict( svmFinalModel,E1(:,I)); 
      C1 = confusionmat(double(groupsallE),double(labelE));
     [x1_e,y1_e,~,auc1] = perfcurve(double(groupsallE),scoreE(:,2),1);
    
