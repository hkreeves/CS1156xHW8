##
## EdX - Machine Learning
## HW8
##
## Author: Kai He
##

library(LowRankQP)

## load data
hw8.loadData <- function()
{
	train <- read.table("http://www.amlbook.com/data/zip/features.train", header=F,
				col.names=c("digit", "symmetry", "intensity"))
	test <- read.table("http://www.amlbook.com/data/zip/features.test", header=F,
				col.names=c("digit", "symmetry", "intensity"))
	return(list(train=train, test=test))
}

## overviews of digit 1 in train set
#dig1 <- train[train$digit == 1, ]
#smoothScatter(dig1$symmetry, dig1$intensity)

## assign class to data
## if ref2 is not specified, digit=ref1 get +1 and all other digits get -1
## if ref2 is specified, digit=ref1 get +1, and digit=ref2 get -1, all others
## are dropped
getClass <- function(data, ref1, ref2=-1)
{ 
	if(ref2 == -1)
	{
		#X <- data[, c(2,3)]
		#Y <- rep(-1, nrow(data))
		#Y[data$digit==ref1] <- 1
		data2 <- data
		data2$digit <- -1
		data2$digit[data$digit==ref1] <- 1
	}
	else
	{
		data2 <- data[(data$digit==ref1) | (data$digit==ref2),]
		#X <- data[, c(2,3)]
		#Y <- rep(-1, nrow(data))
		#Y[data$digit==ref1] <- 1
		data2$digit[data2$digit==ref2] <- -1
		data2$digit[data2$digit==ref1] <- 1
	}
	#return(list(X=as.matrix(X), Y=Y))
	return(data2)
}

## svm with C and Q-degree polynomial kernel 
svm.poly <- function(X, Y, C, Q)
{
	# solve the quadratic programing problem with SVM
	# in the dual form min(-d^T b + 1/2 b^T D b)
	# with the constraints A^T b >= b_0
	# Our problem has
	# constrants 0 =< alpha_n( <= inf), and Y^T alpha = 0
	
	#X <- as.matrix(X)
	Dmat <- (Y %*% t(Y)) * (1+(X %*% t(X)))^Q
	dvec <- rep(-1, nrow(X))
	Amat <- t(Y)
	bvec <- 0
	uvec <- rep(C, nrow(X)) # C is set 1000, should be theoretically inf

	res <- LowRankQP(Dmat, dvec, Amat, bvec, uvec, method="CHOL")
	alphas <- res$alpha
	#print(res$alpha)
	sup.vec <- which(alphas > 1e-8)
	sup.alphas <- alphas[sup.vec]
	margin <- which(alphas < C - 1e-8 & alphas > 1e-8) ## get the margin support vectors
	b <- Y[margin] - t(sup.alphas*Y[sup.vec]) %*% 
		(1+(X[sup.vec,] %*% t(X[margin,])))^Q
 
	## w <- t(X[sup.vec,]) %*% (alphas[sup.vec] * Y[sup.vec])
	#print(w)
	## b <- Y[sup.vec] - X[sup.vec, ] %*% w
	print(as.vector(b))
	print(alphas[sup.vec,])

	return(list(sup.vec=sup.vec, alphas=sup.alphas, b=mean(b)))
}

eval.poly <- function(X, Y, svm.res, Q)
{
	sup.vec <- svm.res$sup.vec
	alphas <- svm.res$alphas
	b <- svm.res$b
	Yp <- sign( t(alphas*Y[sup.vec]) %*% 
			(1+(X[sup.vec,] %*% t(X)))^Q + b )
	return(mean(Yp != Y))
}

## using "e1071" package
library(e1071)

applySVM <- function(ref1, ref2=-1, C, Q)
{
	train2 <- getClass(train, ref1, ref2)
	test2 <- getClass(test, ref1, ref2)
	
	res <- svm(digit ~ symmetry + intensity, data=train2, scale=F, 
			type="C-classification", 
			kernel="polynomial", 
			coef0=1, gamma=1, degree=Q, cost=C, 
			tol=1e-3)
	summary(res)
	
	## E_in
	print(mean(train2$digit != res$fitted))
	## E_out
	print( mean(test2$digit != predict(res, test2[,2:3])) )

	#plot(res, data=train2, digit ~ symmetry + intensity)

	return(res)
}

## Q7-8
## grid search for best C using cross validation
cvSVM <- function(ref1, ref2=-1, Cs, Q, cross, trial)
{
	train2 <- getClass(train, ref1, ref2)
	test2 <- getClass(test, ref1, ref2)

	## sort C ascendingly so to favor the smaller value in a tie
	Cs <- sort(Cs)
	winner <- rep(0, length(Cs))
	acc <- vector(length=length(Cs))
	names(acc) <- Cs
	names(winner) <- Cs
	accMat <- vector()

	for(i in 1:trial)
	{
		## I permutate the data matrix to get different validation splits
		## It can also be done by setting seed option in svm()
		permu <- sample(nrow(train2),replace=F)
		for(C in Cs)
		{	
			res <- svm(digit ~ symmetry + intensity, data=train2[permu,], scale=F, 
			type="C-classification", 
			kernel="polynomial", 
			coef0=1, gamma=1, degree=Q, cost=C, 
			cross=cross, tol=1e-3)
			acc[as.character(C)] <- res$tot.accuracy
		}
		#print(acc)
		accMat <- rbind(accMat, acc)
		winner[which.max(acc)] <- winner[which.max(acc)] + 1
	}
	print(winner)
	colnames(accMat) <- Cs
	rownames(accMat) <- 1:trial
	return(accMat)
}

applyRBF <- function(ref1, ref2=-1, C, gamma)
{
	train2 <- getClass(train, ref1, ref2)
	test2 <- getClass(test, ref1, ref2)
	res <- svm(digit ~ symmetry + intensity, data=train2, scale=F, 
			type="C-classification", 
			kernel="radial",
			gamma=1, cost=C)
	print(res)
	## E_in
	print(mean(train2$digit != res$fitted))
	## E_out
	print( mean(test2$digit != predict(res, test2[,2:3])) )

	res
}
