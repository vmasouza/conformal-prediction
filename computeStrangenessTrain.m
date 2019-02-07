function [strangeness_train, pValues_train] = computeStrangenessTrain(train_data, train_labels, k)
    
    
    classes = tabulate(train_labels);
    classes = classes(:,1);
    
    nClass = length(classes);
    
    allDists = pdist(train_data, 'euclidean');
    distanceMatrix = squareform(allDists);
    Q = eye(size(distanceMatrix))==1;
    distanceMatrix(Q) = Inf;
    
    %       calcula strangess dos exemplos do treino
    strangeness_train = [];
    for c = 1:nClass
        idx_pos = find(train_labels==classes(c));
        idx_neg = find(train_labels~=classes(c));
        tmp_values = [];
        for i = 1:size(distanceMatrix,1)
            
            distPos = sort(distanceMatrix(i, idx_pos), 'ascend');
            if length(distPos) >= k %caso existam menos exemplos da mesma classe do que vizinhos
                distPos = sum(distPos(1:k));
            else
                distPos = sum(distPos(:));
            end
            

            distNeg = sort(distanceMatrix(i, idx_neg), 'ascend');
            if length(distNeg) >= k %caso existam menos exemplos da mesma classe do que vizinhos
                distNeg = sum(distNeg(1:k));
            else
                distNeg = sum(distNeg(:));
            end
            
            
            tmp_values = [tmp_values; distPos/distNeg];
        end
        strangeness_train = [strangeness_train, tmp_values];
    end
    
    %       calcula os p_values dos exemplos do treino
    pValues_train = [];
    for c = 1:nClass
      tmp_pValues = [];
      for i = 1:length(strangeness_train) 
          current_strangeness = strangeness_train(i,c); 
          tmp_strangeness = strangeness_train(:,c); %copia todos os valores de strangeness de uma class
          tmp_strangeness(i) = [];                 %remove o exemplo que esta sendo calculado
          tmp_pValues = [tmp_pValues; (sum(tmp_strangeness > current_strangeness)... 
              + rand(1)*sum(current_strangeness == tmp_strangeness))/length(tmp_strangeness)];
      end
      pValues_train = [pValues_train, tmp_pValues];
    end
    