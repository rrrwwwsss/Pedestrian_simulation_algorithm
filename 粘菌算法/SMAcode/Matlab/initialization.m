
%% ��Ⱥ��ʼ������
function X = initialization(pop,ub,lb,dim)
    %pop:Ϊ��Ⱥ����
    %dim:ÿ�������ά��
    %ub:Ϊÿ��ά�ȵı����ϱ߽磬ά��Ϊ[1,dim];
    %lb:Ϊÿ��ά�ȵı����±߽磬ά��Ϊ[1,dim];
    %X:Ϊ�������Ⱥ��ά��[pop,dim];
    X = zeros(pop,dim); %ΪX���ȷ���ռ�
    for i = 1:pop
       for j = 1:dim
           X(i,j) = (ub(j) - lb(j))*rand() + lb(j);  %����[lb,ub]֮��������
       end
    end
end
