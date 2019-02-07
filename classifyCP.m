
function [vet_bin_acc, martingales, alarm, confidences, all_pValues, strangeness_test] = classifyCP(train_data, train_labels, test_data, test_labels, k)
    

    classes = tabulate(train_labels);
    classes = classes(:,1);
    vet_bin_acc = [];
    confidences = [];
    all_pValues = [];

    [strangeness_train, ~]  = computeStrangenessTrain(train_data, train_labels, k);
    strangeness_test = [];
    alarm = [];

    % martingale parameters
    lambda = 20;
    e = 0.92; % PQ? Pq deus quer...
    martingales = zeros(1,length(classes))+1; % M(0) = 1 for all martingales
    
    for ts = 1:length(test_data)
        current_example = test_data(ts,:);
        actual_label = test_labels(ts);
        
        %calculate strangeness and p-values of test example
        [strang, p_values_test] = computeStrangenessExample(current_example, strangeness_train, train_data, train_labels, k);
        strangeness_test = [strangeness_test; strang];
        all_pValues = [all_pValues; p_values_test];
        
        % calculate martingales values
        current_martingale = [];
        for c = 1:size(p_values_test,2) %calculate martingale value for each class
            current_martingale = [current_martingale, e*p_values_test(c)^(e-1)];
        end
        
        % update martingale values
        M = current_martingale(1,:).*martingales(end, :);
        if max(M) > lambda
            alarm = [alarm; ts, lambda/2];
            martingales = [martingales; zeros(1,length(p_values_test))+1]; 
        else
            martingales = [martingales; current_martingale(1,:).*martingales(end, :)];
        end
        
        
        %classify test example
        [conf, pos] = max(p_values_test);
        predicted_label = classes(pos);
        confidences = [confidences; conf, predicted_label, actual_label];
        
        if predicted_label == actual_label
            vet_bin_acc = [vet_bin_acc; 1]; % acertou
        else
            vet_bin_acc = [vet_bin_acc; 0]; % errou
        end
        
    end
    
    for p = 1 : length(classes)
        subplot(length(classes),1,p); 
        plot(martingales(:,p)); 
        xlabel('Data stream'); 
        ylabel('Martingale value'); 
        hold on; 
        plot(alarm(:,1), alarm(:,2), '*r'); 
        hold on; 
        axis([0 length(vet_bin_acc) 0 lambda]);
    end
end