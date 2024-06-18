using Statistics

B,T,C  = 4,8,2
x = randn(B,T,C)

mean(x[1,1:3,:])