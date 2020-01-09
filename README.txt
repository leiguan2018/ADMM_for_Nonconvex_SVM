代码所用的数据集都是libsvm数据集

main.m使用交叉验证确定参数
main1.m用超参数测试正确率
2017/12/21

运行本程序时，先运行cv.m，通过交叉验证选择最佳rho1和rho2

然后用选择的最佳参数运行main.m获得正确率和运行时间

final version/2017/12/22

cv.m 用十折交叉验证选择对串行ADMM对真实数据集选择超参数rho1 和 rho2
cv_gen.m 用十折交叉验证选择对串行ADMM对生成数据集选择超参数rho1 和 rho2

cv_lambda.m 用十折交叉验证对scadsvm和gist选择真实数据集的超参数lambda
cv_gen_lambda.m 用十折交叉验证对scadsvm和gist选择生成数据的超参数lambda