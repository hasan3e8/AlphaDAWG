function Transform(num_,p,q)
    if p==1
        S='sweep_';
    else S='neut_';
    end
    if q==1
        R='test_';
    else R='train_';
    end
    X1 = readtable(strcat('/Users/fhasan2020/Downloads/T-REx-master/Data/CSV_files/',S,R,'Processed_1.csv'));
    X1(:,1) = [];
    X1(1,:) = [];
    X = table2array(X1);
    C = fdct_wrapping(X,1);
    A = [];
    for i = 1:length(C)
        for j = 1:length(C{i})
            D = reshape(C{i}{j}, 1, []);
            A  = [A D];
        end
    end
    writematrix(A,strcat('/Users/fhasan2020/Downloads/T-REx-master/Data/Curvelets_',S,R,'.csv'));
    for u=2:num_
        X1 = readtable(strcat('/Users/fhasan2020/Downloads/T-REx-master/Data/CSV_files/',S,R,'Processed_', num2str(u), '.csv'));
        X1(:,1) = [];
        X1(1,:) = [];
        X = table2array(X1);
        C = fdct_wrapping(X,1);
        A = [];
        for i = 1:length(C)
           for j = 1:length(C{i})
               D = reshape(C{i}{j}, 1, []);
               A  = [A D];
           end
        end
        writematrix(A,strcat('/Users/fhasan2020/Downloads/T-REx-master/Data/Curvelets_',S,R,'.csv'),'WriteMode','append');
    end
end