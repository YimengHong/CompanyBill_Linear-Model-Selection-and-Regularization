# Load data.
CompanyBill=read.table(".../CompanyBill.txt",header=TRUE);
CompanyBill=CompanyBill[-1,];
# Delete the rows having missing data.
CompanyBill[CompanyBill==0]=NA;
CompanyBill=na.omit(CompanyBill);
# Check dimension
dim(CompanyBill)# [1] 4266    7

install.packages("leaps");
library(leaps);
# Rename the 7 candidate predictors.
colnames(CompanyBill)=c("V1","V2","V3","V4","V5","V6","V7");
# Perform best subset selection.
regfit.full=regsubsets(V1 ~.,CompanyBill);
summary(regfit.full);

# Create a function to illustrate the statistics for subset selection.
Statistic=function (n) 
{par(mfrow =c(2,2));
  plot(reg.summary$rsq ,xlab=" Number of Variables ",ylab=" RSQ", type="l");
  plot(reg.summary$adjr2 ,xlab =" Number of Variables ", ylab=" Adjusted RSQ",type="l");
  a1=which.max(reg.summary$adjr2);
  points(a1,reg.summary$adjr2[a1], col ="red",cex =2, pch =20);
  plot(reg.summary$cp,xlab=" Number of Variables ",ylab="Cp", type="l");
  a2=which.min(reg.summary$cp);
  points(a2,reg.summary$cp[a2], col="red",cex =2, pch =20);
  plot(reg.summary$bic ,xlab=" Number of Variables ",ylab=" BIC", type="l");
  a3=which.min(reg.summary$bic);
  points(a3, reg.summary$bic[a3], col="red",cex=2,pch =20);
}

# Plot the statistics.
Statistic(10000);
# We see the best choice is 3 variables. And the best 3-variable model contains V2, V3 and V5.
coef(regfit.full,3);
# Use forward step subset selection. 
regfit.fwd=regsubsets (V1~.,data=CompanyBill,method="forward");summary(regfit.fwd)
# We see the results are the same as best subset method.

# Choose half of the observed data as a training set
train=sample (1: nrow(x), nrow(x)/2);

# Perform ridge regression on V1~ V2 to V7 on the training set. Use cross-validation to find the best lambda. Show the coefficients when lambda is the best one.

# Install packages for ridge and the lasso.
install.packages("glmnet");
library (glmnet);
# Candidate predictors.
x=model.matrix (V1~.,CompanyBill)[,-1];
# Responses.
y=CompanyBill$V1;
set.seed (1);
train=sample (1: nrow(x), nrow(x)/2);
test=(- train );
cv.out =cv.glmnet (x[train ,],y[train],alpha =0);
# Best lambda found through 10-fold cross-validation (by default).
cv.out$lambda.min;# [1] 45888.09
# Use this lambda to run ridge.
ridge.mod =glmnet(x,y,alpha =0,lambda=45888.09);
coef(ridge.mod)

# Perform the Lasso regression on V1~V2 to V7 on the same training set.
# Determine the best lambda, through 10-fold cross-validation.
set.seed(1);
cv.out1 =cv.glmnet (x[train ,],y[train],alpha =1);
cv.out1$lambda.min;# [1]  3019.13
# Use this lambda to run the lasso.
lasso.mod =glmnet(x,y,alpha =1,lambda=3019.13);
coef(lasso.mod)
# We see the lasso cancel 2 variables V4 and V6.

# Use validation method to tell which model is better between ridge and the Lasso.
ridge.pred=predict (ridge.mod ,s=45888.09, newx=x[test,]);
y.test=y[test];
# MSE for ridge.
mean(( ridge.pred -y.test)^2);# [1]  5443143464
# MSE for the lasso.
lasso.pred=predict (lasso.mod ,s=3019.13, newx=x[test,]);
mean(( lasso.pred -y.test)^2);# [1]  3365049175

# Run tree-based method.
install.packages("tree");
library(tree);
tree.bill=tree(V1~.,CompanyBill, subset=train);
tree.predict=predict (tree.bill,newdata=CompanyBill[-train ,]);

# MSE of tree.
mean((tree.predict-y.test)^2)# [1]  48208606819

# We conclude that in this project the lasso does the best job in prediction: the lasso > ridge > tree.
