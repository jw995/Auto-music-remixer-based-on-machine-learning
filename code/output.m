clc;
clear;
close all hidden;

k=load('output.txt');
%k=k.y;
k=k.*10^2;
k=smooth(k);

audiowrite('../result/2.wav',k,44100/4);