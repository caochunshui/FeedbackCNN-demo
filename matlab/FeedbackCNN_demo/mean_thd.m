function [ y] = mean_thd( x,t )
%MEAN_THD Summary of this function goes here
%   Detailed explanation goes here
tmp=x(find(x>0));
m_d=mean(tmp(:));
x(find(x<t*m_d))=0;
y=x;
end

