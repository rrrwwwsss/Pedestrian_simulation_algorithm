% 黏菌优化算法
% % 微信公众号：KAU的云实验台

function [Convergence_curve,Destination_fitness,bestPositions]=SMA(N,Max_iter,ub,lb,dim,fobj)


%% 参数初始化
bestPositions=zeros(1,dim);
Destination_fitness=inf;%change this to -inf for maximization problems
AllFitness = inf*ones(N,1);%record the fitness of all slime mold
weight = ones(N,dim);%fitness weight of each slime mold

%% 种群初始化
X=initialization(N,ub,lb,dim);
Convergence_curve=zeros(1,Max_iter);
it=1;  %Number of iterations
z=0.03; % parameter

%% 迭代
while  it <= Max_iter
    
    % 适应度计算，边界限制
    for i=1:N
        % Check if solutions go outside the search space and bring them back
        Flag4ub=X(i,:)>ub;
        Flag4lb=X(i,:)<lb;
        X(i,:)=(X(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        AllFitness(i) = fobj(X(i,:));
    end
    
    % 适应度排序
    [SmellOrder,SmellIndex] = sort(AllFitness);  
    worstFitness = SmellOrder(N);
    bestFitness = SmellOrder(1);

    S=bestFitness-worstFitness+eps;  % plus eps to avoid denominator zero

    % 更新黏菌重量
    for i=1:N
        for j=1:dim
            if i<=(N/2) 
                weight(SmellIndex(i),j) = 1+rand()*log10((bestFitness-SmellOrder(i))/(S)+1);
            else
                weight(SmellIndex(i),j) = 1-rand()*log10((bestFitness-SmellOrder(i))/(S)+1);
            end
        end
    end
    
    % 最优更新
    if bestFitness < Destination_fitness
        bestPositions=X(SmellIndex(1),:);
        Destination_fitness = bestFitness;
    end
    
    % 下式中的a、b分别是vb和vc中的参数
    a = atanh(-(it/Max_iter)+1);  
    b = 1-it/Max_iter;

    % 核心更新机制
    for i=1:N

        if rand<z    
            % 分离部分个体搜索其他食物
            X(i,:) = (ub-lb)*rand+lb;
        else
            % 搜索食物
            p =tanh(abs(AllFitness(i)-Destination_fitness));  
            vb = unifrnd(-a,a,1,dim);
            vc = unifrnd(-b,b,1,dim);
            for j=1:dim
                r = rand();
                A = randi([1,N]);  % two positions randomly selected from population
                B = randi([1,N]);
                if r<p 
                    X(i,j) = bestPositions(j)+ vb(j)*(weight(i,j)*X(A,j)-X(B,j));
                else
                    X(i,j) = vc(j)*X(i,j);
                end
            end
        end
    end
    Convergence_curve(it)=Destination_fitness;
    it=it+1;
end

end
