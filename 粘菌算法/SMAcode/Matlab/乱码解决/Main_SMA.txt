
% ----------------------- README ------------------------------------------
% -------------- 最后一次修改：2023/12/17 ----------------------------------
% -------------------  欢迎关注₍^.^₎♡  ------------------------------------
% -------------- 项目：黏菌算法(SMA)函数寻优  -------------------------------
% -------------- 微信公众号：KAU的云实验台(可咨询定制) -----------------------
% -------------- CSDN：KAU的云实验台 ---------------------------------------
% -------------- 付费代码(更全)：https://mbd.pub/o/author-a2iWlGtpZA== -----
% -------------- 免费代码：公众号后台回复"资源" -----------------------------
% -------------------------------------------------------------------------

%% 释放空间
clc;
clear;
close all;

%% 参数设置
SearchAgents_no=30; % 种群量级
Function_name='F15'; % 测试函数
Max_iteration=500; % 迭代次数
[lb,ub,dim,fobj]=Get_Functions_details(Function_name); % 获取测试函数上下界、维度、函数表达式
lb= lb.*ones( 1,dim );
ub= ub.*ones( 1,dim );

%% 黏菌算法求解
[IterCurve,Best_fitness,Best_Pos] = SMA(SearchAgents_no,Max_iteration,ub,lb,dim,fobj);

%% 结果
figure('Position',[269   240   660   290])
subplot(1,2,1);
func_plot(Function_name);
title('Parameter space')
xlabel('x_1');
ylabel('x_2');
zlabel([Function_name,'( x_1 , x_2 )'])

subplot(1,2,2);
plot(IterCurve,'Color','r','LineWidth',2)
title('Objective space')
xlabel('Iteration');
ylabel('Best score obtained so far');
axis tight
grid on
box on
legend('SMA')
