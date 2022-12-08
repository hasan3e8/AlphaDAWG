function EMP_Transform_Curvelet(num_)
    X1 = readtable(strcat('../Data/CSV_files/output_1.csv'));
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
    writematrix(A,strcat('../Data/EMP_Curvelets_.csv'));
    for u=2:num_
        X1 = readtable(strcat('../Data/CSV_files/output_', num2str(u), '.csv'));
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
        writematrix(A,strcat('../Data/EMP_Curvelets_.csv'),'WriteMode','append');
    end
end