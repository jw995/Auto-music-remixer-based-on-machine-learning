clc;
clear;

close all hidden;

[y,Fs] = audioread('../data/selected/attention.wav');
[y1,Fs1] = audioread('../data/selected/attention1.wav');
m=size(y,1);
m1=size(y1,1);

diff=m-m1;
comeon=zeros(diff,2);
y1=[y1;comeon];
diff_y=y-y1;

y=downsample(y,4);
y=[smooth(y(:,1)) smooth(y(:,2))];
y1=downsample(y1,4);
y1=[smooth(y1(:,1)) smooth(y1(:,2))];


audiowrite('../result/1.wav',y,Fs/4);

% save('../result/attention.mat','y','y1')

% cutting the matrix into pieces 
n=10;
   unit=int32(size(y,1)/n);
for i=1:n
    start=(i-1)*unit+1;
    endding=i*unit;
    y2=y(start:endding,:);
    filename=['attention_',num2str(i),'.csv'];
    
    y3=y1(start:endding,:);
    filename1=['attention1_',num2str(i),'.csv'];
    csvwrite(filename,y2);
    csvwrite(filename1,y3);
end






