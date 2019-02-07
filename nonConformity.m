function [strangeness_train, strangeness_test_example, p_values_train, p_values_test] = nonConformity(train_data, train_labels, k, test_example)


    classes = tabulate(train_labels);
    classes = classes(:,1);
    
    nClass = length(classes);
    
    allDists = pdist([train_data; test_example], 'euclidean');
    distanceMatrix = squareform(allDists);
    Q = eye(size(distanceMatrix))==1;
    distanceMatrix(Q) = Inf;

    strangeness_train = [];
    strangeness_test_example = [];

    for c = 1:nClass
        idx_pos = find(train_labels==classes(c));
        idx_neg = find(train_labels~=classes(c));
        temp_values = [];
        for i = 1:size(distanceMatrix,1)-1
            distPos = sort(distanceMatrix(i, idx_pos), 'ascend');
            distPos = sum(distPos(1:k));

            distNeg = sort(distanceMatrix(i, idx_neg), 'ascend');
            distNeg = sum(distNeg(1:k));
            
            temp_values = [temp_values; distPos/distNeg];
        end
        strangeness_train = [strangeness_train, temp_values];
        
        % calcula strangeness p/ exemplo de teste
        distPos_test = sort(distanceMatrix(end, idx_pos), 'ascend');
        distPos_test = sum(distPos_test(1:k));
        
        distNeg_test = sort(distanceMatrix(end, idx_neg), 'ascend');
        distNeg_test = sum(distNeg_test(1:k));
        
        strangeness_test_example = [strangeness_test_example, ...
            (distPos_test/distNeg_test)];
    end
    
    p_values_train = [];
    p_values_test = [];
    for c = 1:nClass

       %         calcula os p_values do exemplo de teste
        p_values_test = [p_values_test, (sum(strangeness_train(:,c) > strangeness_test_example(c))... 
          + rand(1)*sum(strangeness_test_example(c) == strangeness_train(:,c)))/length(strangeness_train)];

      %       calcula os p_values dos exemplos do treino
      temp_pValues = [];
      for i = 1:length(strangeness_train) 
          actual_strangeness = strangeness_train(i,c); 
          temp_strangeness = strangeness_train(:,c); %copia todos os valores de strangeness de uma class
          temp_strangeness(i) = [];                 %remove o exemplo que esta sendo calculado
          temp_pValues = [temp_pValues; (sum(temp_strangeness > actual_strangeness)... 
              + rand(1)*sum(actual_strangeness == temp_strangeness))/length(temp_strangeness)];
      end
      p_values_train = [p_values_train, temp_pValues];
      
    end

    
    