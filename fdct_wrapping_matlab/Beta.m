X1 = readtable('/Users/fhasan2020/Desktop/NNs/NN Const Hard/Beta for C to be processed_8.csv');
T = X1(:,2);
X = table2array(T);
%X(1)= [];

S = readtable('ABoutput_0.csv');
S(:,1) = [];
S1 = table2array(S);
C = fdct_wrapping(S1,1);  
%A = [];
%for i = 1:length(C)
%    for j = 1:length(C{i})
%        D = reshape(C{i}{j}, 1, []);
%        A  = [A D];
%    end
%    length(A)
%end

o=1;
A =  [];
for i = 1:length(C)
     for j = 1:length(C{i})
         for k = 1:size(C{i}{j},2)
             for l = 1:size(C{i}{j},1)
                 C{i}{j}(l,k) = X(o);
                 o=o+1;
             end
         end
     end
end

Y = ifdct_wrapping(C,1);
%heatmap(Y.')
size(Y)
%writematrix(Y,strcat('/Users/fhasan2020/Desktop/NNs/NN Const Hard/Beta for C from CW_8.csv'));
