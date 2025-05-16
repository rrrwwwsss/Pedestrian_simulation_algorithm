
%% 种群初始化函数
function X = initialization(pop,ub,lb,dim)
    %pop:为种群数量
    %dim:每个个体的维度
    %ub:为每个维度的变量上边界，维度为[1,dim];
    %lb:为每个维度的变量下边界，维度为[1,dim];
    %X:为输出的种群，维度[pop,dim];
    X = zeros(pop,dim); %为X事先分配空间
    for i = 1:pop
       for j = 1:dim
           X(i,j) = (ub(j) - lb(j))*rand() + lb(j);  %生成[lb,ub]之间的随机数
       end
    end
end
