clear
clc

currentpath = cd;

addpath(genpath([currentpath,'/CFile']));

cd ./CFile

mex proximalRegC.c
mex funRegC.c

cd ..
cd ..