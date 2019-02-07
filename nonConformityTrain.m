function NCScore_train = nonConformityTrain(train_data, train_labels, k, test_example)


    classes = tabulate(train_labels);
    classes = classes(:,1);
    nClass = length(classes);
    allDists = pdist(train_data, 'euclidean');
    distanceMatrix = squareform(allDists);
    Q = eye(size(distanceMatrix))==1;
    distanceMatrix(Q) = Inf;

    NCScore = [];

    
    for i = 1:length(train_data)
        actual_label = train_labels(i);
        idx_pos = find(train_labels==actual_label);
        idx_neg = find(train_labels~=actual_label);

        distPos = sort(distanceMatrix(i, idx_pos), 2);
        distPos = distPos(1:k);

        distNeg = sort(distanceMatrix(i, idx_neg), 2);
        distNeg = distNeg(1:k);
        NCScore_train = [NCScore_train; distPos/distNeg];
    end
    
    NCScore_test = [];
    


% function p_value = calculatePValue(NCScore_Train, unlabeled_data)
    