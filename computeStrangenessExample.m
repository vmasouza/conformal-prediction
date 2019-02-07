function [strangeness_test_example, p_values_test] = computeStrangenessExample(test_example, strangeness_train, train_data, train_labels, k)

    classes = tabulate(train_labels);
    classes = classes(:,1);
    
    nClass = length(classes);
    strangeness_test_example = [];
    p_values_test = [];
    
    % calculate euclidean distance of test example -> train data
    test_distances = [];
    for tr = 1:length(train_labels)
        test_distances = [test_distances, pdist([train_data(tr, :); test_example])];
    end
    
    for c = 1:nClass
        idx_pos = find(train_labels==classes(c));
        idx_neg = find(train_labels~=classes(c));
    
        % calcula strangeness p/ exemplo de teste
        distPos_test = sort(test_distances(idx_pos), 'ascend');
        if length(distPos_test) >= k %caso existam menos exemplos da mesma classe do que vizinhos
            distPos_test = sum(distPos_test(1:k));
        else
            distPos_test = sum(distPos_test(:));
        end

        distNeg_test = sort(test_distances(idx_neg), 'ascend');
        if length(distNeg_test) >= k %caso existam menos exemplos da mesma classe do que vizinhos
            distNeg_test = sum(distNeg_test(1:k));
        else
            distNeg_test = sum(distNeg_test(:));
        end

        strangeness_test_example = [strangeness_test_example, ...
            (distPos_test/distNeg_test)];
        
        
        
       % calcula os p_values do exemplo de teste
        p_values_test = [p_values_test, (sum(strangeness_train(:,c) > strangeness_test_example(c))... 
          + rand(1)*sum(strangeness_test_example(c) == strangeness_train(:,c)))/length(strangeness_train)];
        
    end
end