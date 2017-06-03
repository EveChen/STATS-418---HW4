###1. Setup
#Remove Objects
rm(list=ls())

#Clear Memory
gc(reset=TRUE)

#Set Working Directory
setwd("C:/Users/Eve/Dropbox/UCLA Files/Courses/418 Tools of Data Science/HW4")

#Load packages
library(readr)
library(ROCR)


###2. Method - Neural Network
#(a.) Default Layers (200, 200) with early stopping
library(h2o)
h2o.init(nthreads=-1)

dx <- h2o.importFile("test.csv")

dx_split <- h2o.splitFrame(dx, ratios = c(0.6,0.2), seed = 123)
dx_train <- dx_split[[1]]
dx_valid <- dx_split[[2]]
dx_test <- dx_split[[3]]

Xnames <- names(dx_train)[which(names(dx_train)!="y")]

system.time({
  md1 <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
                         ## DEFAULT: activation = "Rectifier", hidden = c(200,200), 
                         epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
auc1 <- h2o.performance(md1, dx_test)@metrics$AUC
auc1

#(b.)
system.time({
  md2 <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
                         activation = "Rectifier", hidden = c(50,50,50,50), input_dropout_ratio = 0.2,
                         epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
auc2 <- h2o.performance(md2, dx_test)@metrics$AUC
auc2

#(c.)
system.time({
  md3 <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
                         activation = "Rectifier", hidden = c(50,50,50,50), 
                         epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
auc3 <- h2o.performance(md3, dx_test)@metrics$AUC
auc3

#(d.)
system.time({
  md4 <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
                         activation = "Rectifier", hidden = c(20,20),
                         epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
auc4 <- h2o.performance(md4, dx_test)@metrics$AUC
auc4

#(e.)
system.time({
  md5 <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
                         activation = "Rectifier", hidden = c(20),
                         epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
auc5 <- h2o.performance(md5, dx_test)@metrics$AUC
auc5

#(f.)
system.time({
  md6 <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
                         activation = "Rectifier", hidden = c(5),
                         epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
auc6 <- h2o.performance(md6, dx_test)@metrics$AUC
auc6

#(g.)
system.time({
  md7 <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
                         activation = "Rectifier", hidden = c(1),
                         epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
auc7 <- h2o.performance(md7, dx_test)@metrics$AUC
auc7

#(h.)
system.time({
  md8 <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
                         activation = "Rectifier", hidden = c(200,200), l1 = 1e-5, l2 = 1e-5, 
                         epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
auc8 <- h2o.performance(md8, dx_test)@metrics$AUC
auc8

#(i.)
system.time({
  md9 <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
                         activation = "RectifierWithDropout", hidden = c(200,200,200,200), hidden_dropout_ratios=c(0.2,0.1,0.1,0),
                         epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
auc9 <- h2o.performance(md9, dx_test)@metrics$AUC
auc9

#(j.)
system.time({
  md10 <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
                         activation = "Rectifier", hidden = c(200,200), 
                         rho = 0.95, epsilon = 1e-06,  ## default:  rho = 0.99, epsilon = 1e-08
                         epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
auc10 <- h2o.performance(md10, dx_test)@metrics$AUC
auc10

#(k.)
system.time({
  md11 <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
                         activation = "Rectifier", hidden = c(200,200), 
                         rho = 0.999, epsilon = 1e-08,  ## default:  rho = 0.99, epsilon = 1e-08
                         epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
auc11 <- h2o.performance(md11, dx_test)@metrics$AUC
auc11

#(l.)
system.time({
  md12 <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
                         activation = "Rectifier", hidden = c(200,200), 
                         rho = 0.9999, epsilon = 1e-08,  ## default:  rho = 0.99, epsilon = 1e-08
                         epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
auc12 <- h2o.performance(md12, dx_test)@metrics$AUC
auc12

#(m.)
system.time({
  md13 <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
                         activation = "Rectifier", hidden = c(200,200), 
                         rho = 0.999, epsilon = 1e-06,  ## default:  rho = 0.99, epsilon = 1e-08
                         epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
auc13 <- h2o.performance(md13, dx_test)@metrics$AUC
auc13

#(n.)
system.time({
  md14 <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
                         activation = "Rectifier", hidden = c(200,200), 
                         rho = 0.999, epsilon = 1e-09,  ## default:  rho = 0.99, epsilon = 1e-08
                         epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
auc14 <- h2o.performance(md14, dx_test)@metrics$AUC
auc14

#(o.)
system.time({
  md15 <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
                         activation = "Rectifier", hidden = c(200,200), 
                         adaptive_rate = FALSE, ## default: rate = 0.005, rate_decay = 1, momentum_stable = 0,
                         epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
auc15 <- h2o.performance(md15, dx_test)@metrics$AUC
auc15 

#(p.)
system.time({
  md16 <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
                         activation = "Rectifier", hidden = c(200,200), 
                         adaptive_rate = FALSE, rate = 0.001, momentum_start = 0.5, momentum_ramp = 1e5, momentum_stable = 0.99,
                         epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
auc16 <- h2o.performance(md16, dx_test)@metrics$AUC
auc16 

#(q.)
system.time({
  md17 <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
                         activation = "Rectifier", hidden = c(200,200), 
                         adaptive_rate = FALSE, rate = 0.01, momentum_start = 0.5, momentum_ramp = 1e5, momentum_stable = 0.99,
                         epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
auc17 <- h2o.performance(md17, dx_test)@metrics$AUC
auc17

#(r.)
system.time({
  md18 <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
                         activation = "Rectifier", hidden = c(200,200), 
                         adaptive_rate = FALSE, rate = 0.01, rate_annealing = 1e-05, 
                         momentum_start = 0.5, momentum_ramp = 1e5, momentum_stable = 0.99,
                         epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
auc18 <- h2o.performance(md18, dx_test)@metrics$AUC
auc18

#(s.)
system.time({
  md19 <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
                         activation = "Rectifier", hidden = c(200,200), 
                         adaptive_rate = FALSE, rate = 0.01, rate_annealing = 1e-04, 
                         momentum_start = 0.5, momentum_ramp = 1e5, momentum_stable = 0.99,
                         epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
auc19 <- h2o.performance(md19, dx_test)@metrics$AUC
auc19

#(t.)
system.time({
  md20 <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
                         activation = "Rectifier", hidden = c(200,200), 
                         adaptive_rate = FALSE, rate = 0.01, rate_annealing = 1e-05, 
                         momentum_start = 0.5, momentum_ramp = 1e5, momentum_stable = 0.9,
                         epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
auc20 <- h2o.performance(md20, dx_test)@metrics$AUC
auc20

#(u.)
system.time({
  md21 <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
                         activation = "Rectifier", hidden = c(200,200), 
                         adaptive_rate = FALSE, rate = 0.01, rate_annealing = 1e-05, 
                         momentum_start = 0.5, momentum_ramp = 1e4, momentum_stable = 0.9,
                         epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
auc21 <- h2o.performance(md21, dx_test)@metrics$AUC
auc21



#############Don't run
###3. Method - Random Search (with algorithm "GBM")
library(h2o)
h2o.init(nthreads=-1)

dx <- h2o.importFile("test.csv")
dx_split <- h2o.splitFrame(dx, ratios = c(0.6,0.2), seed = 123)
dx_train <- dx_split[[1]]
dx_valid <- dx_split[[2]]
dx_test <- dx_split[[3]]

Xnames <- names(dx_train)[which(names(dx_train)!="y")]

hyper_params <- list( ntrees = 10000,  ## early stopping
                      max_depth = 5:15, 
                      min_rows = c(1,3,10,30,100),
                      learn_rate = c(0.01,0.03,0.1),  
                      learn_rate_annealing = c(0.99,0.995,1,1),
                      sample_rate = c(0.4,0.7,1,1),
                      col_sample_rate = c(0.7,1,1),
                      nbins = c(30,100,300),
                      nbins_cats = c(64,256,1024)
)

search_criteria <- list( strategy = "RandomDiscrete",
                         max_runtime_secs = 10*3600,
                         max_models = 100
)

system.time({
  md22 <- h2o.grid(algorithm = "gbm", grid_id = "grd",
                  x = Xnames, y = "y", training_frame = dx_train,
                  validation_frame = dx_valid,
                  hyper_params = hyper_params,
                  search_criteria = search_criteria,
                  stopping_metric = "AUC", stopping_tolerance = 1e-3, stopping_rounds = 2,
                  seed = 123)
})

md22_sort <- h2o.getGrid(grid_id = "grd", sort_by = "auc", decreasing = TRUE)
md22_sort

md22_best <- h2o.getModel(md22_sort@model_ids[[1]])
summary(md22_best)

auc22 <- h2o.auc(h2o.performance(md22_best, dx_test))
auc22


#4. Method - Ensemble
library(h2o)
h2o.init(nthreads=-1)

dx <- h2o.importFile("test.csv")
dx_split <- h2o.splitFrame(dx, ratios = 0.7, seed = 123)
dx_train <- dx_split[[1]]
dx_test <- dx_split[[2]]
Xnames <- setdiff(names(dx_train),"y")


system.time({
  md23 <- h2o.glm(x = Xnames, y = "y", training_frame = dx_train, 
                 family = "binomial", 
                 alpha = 1, lambda = 0,
                 seed = 123,
                 nfolds = 5, fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE)
})

system.time({
  md24 <- h2o.randomForest(x = Xnames, y = "y", training_frame = dx_train, 
                          ntrees = 300,
                          seed = 123,
                          nfolds = 5, fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE)
})


system.time({
  md25 <- h2o.gbm(x = Xnames, y = "y", training_frame = dx_train, distribution = "bernoulli", 
                 ntrees = 200, max_depth = 10, learn_rate = 0.1, 
                 nbins = 100, seed = 123,
                 nfolds = 5, fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE)    
})

system.time({
  md26 <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, 
                          epochs = 5,
                          seed = 123,
                          nfolds = 5, fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE) 
})



md_ensemble <- h2o.stackedEnsemble(x = Xnames, y = "y", training_frame = dx_train, 
                              base_models = list(md23@model_id, md24@model_id, md25@model_id, md26@model_id))


h2o.auc(h2o.performance(md23, dx_test))
h2o.auc(h2o.performance(md24, dx_test))
h2o.auc(h2o.performance(md25, dx_test))
h2o.auc(h2o.performance(md26, dx_test))
h2o.auc(h2o.performance(md_ensemble, dx_test))


auc_ensemble <- h2o.getModel(md_ensemble@model$metalearner$name)@model$coefficients_table
s

##HW3
###2. Method - Logistic Regression
#Try logistic regression by using glmnet function in R
library(readr)
library(glmnet)
library(ROCR)

d <- read_csv("test.csv")

set.seed(123)
N <- nrow(d)
idx <- sample(1:N, 0.6*N)
d_train <- d[idx,]
d_test <- d[-idx,]


X <- Matrix::sparse.model.matrix(y ~ . - 1, data = d)
X_train <- X[idx,]
X_test <- X[-idx,]

#With lasso + Regularization
system.time({
  md_LR_reg <- glmnet( X_train, d_train$y, lambda = 0, alpha = 1, family = "binomial")
})

#Predict & AUC
phat <- predict(md_LR_reg, newx = X_test, type = "response")
LR_reg_pred <- prediction(phat, d_test$y)
LR_reg_auc <- performance(LR_reg_pred, "auc")@y.values[[1]]

#Without lasso + Regularization
system.time({
  md_LR <- glmnet( X_train, d_train$y, lambda = 1, alpha = 0, family = "binomial")
})

#Predict & AUC
phat <- predict(md_LR, newx = X_test, type = "response")
LR_pred <- prediction(phat, d_test$y)
LR_auc <- performance(LR_pred, "auc")@y.values[[1]]



#Try logistic regression by using h2o packagage within R
library(h2o)
h2o.init(nthreads=-1)

dx <- h2o.importFile("test.csv")

dx_split <- h2o.splitFrame(dx, ratios = 0.6, seed = 123)
dx_train <- dx_split[[1]]
dx_test <- dx_split[[2]]

Xnames <- names(dx_train)[which(names(dx_train)!="y")]

#With lasso + Regularization
system.time({
  md_LR_reg <- h2o.glm(x = Xnames, y = "y", training_frame = dx_train, 
                       family = "binomial", alpha = 1, lambda = 0)
})

#Calculate auc for logistic regression model (using h2o package)
h2o.auc(h2o.performance(md_LR_reg, dx_test))
#See the confusion matrix of our prediction
h2o.confusionMatrix(h2o.performance(md_LR_reg, dx_test))


##Without lasso + Regularization
system.time({
  md_LR <- h2o.glm(x = Xnames, y = "y", training_frame = dx_train, 
                   family = "binomial", alpha = 0, lambda = 1)
})

#Calculate auc for logistic regression model (using h2o package)
h2o.auc(h2o.performance(md_LR, dx_test))
#See the confusion matrix of our prediction
h2o.confusionMatrix(h2o.performance(md_LR, dx_test))




###3. Method - Random Forest
#Try random forest by using h2o packagage within R
library(h2o)

h2o.init(nthreads=-1)

dx <- h2o.importFile("test.csv")

dx_split <- h2o.splitFrame(dx, ratios = 0.6, seed = 123)
dx_train <- dx_split[[1]]
dx_test <- dx_split[[2]]


Xnames <- names(dx_train)[which(names(dx_train)!="y")]

system.time({
  md <- h2o.randomForest(x = Xnames, y = "y", training_frame = dx_train, ntrees = 500)
})


md5 <- h2o.randomForest(x = Xnames, y = "y", training_frame = dx_train, ntrees = 200, nfolds = 0, max_depth = 10)
auc5 <- h2o.auc(h2o.performance(md5, dx_test))
md6 <- h2o.randomForest(x = Xnames, y = "y", training_frame = dx_train, ntrees = 200, nfolds = 0, max_depth = 20)
auc6 <- h2o.auc(h2o.performance(md6, dx_test))
md7 <- h2o.randomForest(x = Xnames, y = "y", training_frame = dx_train, ntrees = 200, nfolds = 0, max_depth = 30)
auc7 <- h2o.auc(h2o.performance(md7, dx_test))

md8 <- h2o.randomForest(x = Xnames, y = "y", training_frame = dx_train, ntrees = 400, nfolds = 0, max_depth = 10)
auc8 <- h2o.auc(h2o.performance(md8, dx_test))
md9 <- h2o.randomForest(x = Xnames, y = "y", training_frame = dx_train, ntrees = 400, nfolds = 0, max_depth = 20)
auc9 <- h2o.auc(h2o.performance(md9, dx_test))
md10 <- h2o.randomForest(x = Xnames, y = "y", training_frame = dx_train, ntrees = 400, nfolds = 0, max_depth = 30)
auc10 <- h2o.auc(h2o.performance(md10, dx_test))

md11 <- h2o.randomForest(x = Xnames, y = "y", training_frame = dx_train, ntrees = 600, nfolds = 0, max_depth = 10)
auc11 <- h2o.auc(h2o.performance(md11, dx_test))
md12 <- h2o.randomForest(x = Xnames, y = "y", training_frame = dx_train, ntrees = 600, nfolds = 0, max_depth = 20)
auc12 <- h2o.auc(h2o.performance(md12, dx_test))
md13 <- h2o.randomForest(x = Xnames, y = "y", training_frame = dx_train, ntrees = 600, nfolds = 0, max_depth = 30)
auc13 <- h2o.auc(h2o.performance(md13, dx_test))



#Try random forest by using xgboost packagage within R
library(readr)
library(xgboost)
library(ROCR)


dd <- read_csv("test.csv")

set.seed(123)
N <- nrow(d)
idx <- sample(1:N, 0.6*N)
dd_train <- dd[idx,]
dd_test <- dd[-idx,]


XX <- Matrix::sparse.model.matrix(y ~ . - 1, data = dd)
XX_train <- XX[idx,]
XX_test <- XX[-idx,]


# random forest with xgboost
system.time({
  n_proc <- parallel::detectCores()
  md14 <- xgboost(data = X_train, label = ifelse(d_train$y=='Y',1,0),
                  nthread = n_proc, nround = 1, max_depth = 20,
                  num_parallel_tree = 500, subsample = 0.632,
                  colsample_bytree = 1/sqrt(length(X_train@x)/nrow(X_train)),
                  save_period = NULL)
})

#14
n_proc <- parallel::detectCores()
md14 <- xgboost(data = XX_train, label = ifelse(d_train$y=='Y',1,0),
                nthread = n_proc, nround = 1, max_depth = 10,
                num_parallel_tree = 200, subsample = 0.632,
                colsample_bytree = 1/sqrt(length(XX_train@x)/nrow(XX_train)),
                save_period = NULL)

phat14 <- predict(md14, newdata = XX_test)
pred14 <- prediction(phat14, dd_test$y)
auc14 <- performance(pred14, "auc")@y.values[[1]]

#15
n_proc <- parallel::detectCores()
md15 <- xgboost(data = XX_train, label = ifelse(d_train$y=='Y',1,0),
                nthread = n_proc, nround = 1, max_depth = 20,
                num_parallel_tree = 200, subsample = 0.632,
                colsample_bytree = 1/sqrt(length(XX_train@x)/nrow(XX_train)),
                save_period = NULL)

phat15 <- predict(md15, newdata = XX_test)
pred15 <- prediction(phat15, dd_test$y)
auc15 <- performance(pred15, "auc")@y.values[[1]]


#16
n_proc <- parallel::detectCores()
md16 <- xgboost(data = XX_train, label = ifelse(d_train$y=='Y',1,0),
                nthread = n_proc, nround = 1, max_depth = 30,
                num_parallel_tree = 200, subsample = 0.632,
                colsample_bytree = 1/sqrt(length(XX_train@x)/nrow(XX_train)),
                save_period = NULL)

phat16 <- predict(md16, newdata = XX_test)
pred16 <- prediction(phat16, dd_test$y)
auc16 <- performance(pred16, "auc")@y.values[[1]]


#17
n_proc <- parallel::detectCores()
md17 <- xgboost(data = XX_train, label = ifelse(d_train$y=='Y',1,0),
                nthread = n_proc, nround = 1, max_depth = 10,
                num_parallel_tree = 400, subsample = 0.632,
                colsample_bytree = 1/sqrt(length(XX_train@x)/nrow(XX_train)),
                save_period = NULL)

phat17 <- predict(md17, newdata = XX_test)
pred17 <- prediction(phat17, dd_test$y)
auc17 <- performance(pred17, "auc")@y.values[[1]]

#18
n_proc <- parallel::detectCores()
md18 <- xgboost(data = XX_train, label = ifelse(d_train$y=='Y',1,0),
                nthread = n_proc, nround = 1, max_depth = 20,
                num_parallel_tree = 400, subsample = 0.632,
                colsample_bytree = 1/sqrt(length(XX_train@x)/nrow(XX_train)),
                save_period = NULL)

phat18 <- predict(md18, newdata = XX_test)
pred18 <- prediction(phat18, dd_test$y)
auc18 <- performance(pred18, "auc")@y.values[[1]]

#19
n_proc <- parallel::detectCores()
md19 <- xgboost(data = XX_train, label = ifelse(d_train$y=='Y',1,0),
                nthread = n_proc, nround = 1, max_depth = 30,
                num_parallel_tree = 400, subsample = 0.632,
                colsample_bytree = 1/sqrt(length(XX_train@x)/nrow(XX_train)),
                save_period = NULL)

phat19 <- predict(md19, newdata = XX_test)
pred19 <- prediction(phat19, dd_test$y)
auc19 <- performance(pred19, "auc")@y.values[[1]]

#20
n_proc <- parallel::detectCores()
md20 <- xgboost(data = XX_train, label = ifelse(d_train$y=='Y',1,0),
                nthread = n_proc, nround = 1, max_depth = 10,
                num_parallel_tree = 600, subsample = 0.632,
                colsample_bytree = 1/sqrt(length(XX_train@x)/nrow(XX_train)),
                save_period = NULL)

phat20 <- predict(md20, newdata = XX_test)
pred20 <- prediction(phat20, dd_test$y)
auc20 <- performance(pred20, "auc")@y.values[[1]]


#21
n_proc <- parallel::detectCores()
md21 <- xgboost(data = XX_train, label = ifelse(d_train$y=='Y',1,0),
                nthread = n_proc, nround = 1, max_depth = 20,
                num_parallel_tree = 600, subsample = 0.632,
                colsample_bytree = 1/sqrt(length(XX_train@x)/nrow(XX_train)),
                save_period = NULL)

phat21 <- predict(md21, newdata = XX_test)
pred21 <- prediction(phat21, dd_test$y)
auc21 <- performance(pred21, "auc")@y.values[[1]]


#22
n_proc <- parallel::detectCores()
md22 <- xgboost(data = XX_train, label = ifelse(d_train$y=='Y',1,0),
                nthread = n_proc, nround = 1, max_depth = 30,
                num_parallel_tree = 600, subsample = 0.632,
                colsample_bytree = 1/sqrt(length(XX_train@x)/nrow(XX_train)),
                save_period = NULL)

phat22 <- predict(md22, newdata = XX_test)
pred22 <- prediction(phat22, dd_test$y)
auc22 <- performance(pred22, "auc")@y.values[[1]]


###4. Method - GBM
#Try GBM by using h20 packagage within R
library(h2o)

h2o.init(nthreads=-1)


dx <- h2o.importFile("test.csv")

dx_split <- h2o.splitFrame(dx, ratios = 0.6, seed = 123)
dx_train <- dx_split[[1]]
dx_test <- dx_split[[2]]


Xnames <- names(dx_train)[which(names(dx_train)!="y")]

system.time({
  md <- h2o.gbm(x = Xnames, y = "y", training_frame = dx_train, distribution = "bernoulli", 
                ntrees = 300, max_depth = 20, learn_rate = 0.1, 
                nbins = 100, seed = 123)    
})

#23
md23 <- h2o.gbm(x = Xnames, y = "y", training_frame = dx_train, distribution = "bernoulli", 
                ntrees = 200, max_depth = 10, learn_rate = 0.01, 
                nbins = 100, seed = 123)

auc23 <- h2o.auc(h2o.performance(md23, dx_test))

#24
md24 <- h2o.gbm(x = Xnames, y = "y", training_frame = dx_train, distribution = "bernoulli", 
                ntrees = 200, max_depth = 10, learn_rate = 0.1, 
                nbins = 100, seed = 123)

auc24 <- h2o.auc(h2o.performance(md24, dx_test))

#25
md25 <- h2o.gbm(x = Xnames, y = "y", training_frame = dx_train, distribution = "bernoulli", 
                ntrees = 200, max_depth = 20, learn_rate = 0.01, 
                nbins = 100, seed = 123)

auc25 <- h2o.auc(h2o.performance(md25, dx_test))

#26
md26 <- h2o.gbm(x = Xnames, y = "y", training_frame = dx_train, distribution = "bernoulli", 
                ntrees = 200, max_depth = 20, learn_rate = 0.1, 
                nbins = 100, seed = 123)

auc26 <- h2o.auc(h2o.performance(md26, dx_test))

#27
md27 <- h2o.gbm(x = Xnames, y = "y", training_frame = dx_train, distribution = "bernoulli", 
                ntrees = 400, max_depth = 10, learn_rate = 0.01, 
                nbins = 100, seed = 123)

auc27 <- h2o.auc(h2o.performance(md27, dx_test))


#28
md28 <- h2o.gbm(x = Xnames, y = "y", training_frame = dx_train, distribution = "bernoulli", 
                ntrees = 400, max_depth = 10, learn_rate = 0.1, 
                nbins = 100, seed = 123)

auc28 <- h2o.auc(h2o.performance(md28, dx_test))


#29
md29 <- h2o.gbm(x = Xnames, y = "y", training_frame = dx_train, distribution = "bernoulli", 
                ntrees = 400, max_depth = 20, learn_rate = 0.01, 
                nbins = 100, seed = 123)

auc29 <- h2o.auc(h2o.performance(md29, dx_test))

#30
md30 <- h2o.gbm(x = Xnames, y = "y", training_frame = dx_train, distribution = "bernoulli", 
                ntrees = 400, max_depth = 20, learn_rate = 0.1, 
                nbins = 100, seed = 123)

auc30 <- h2o.auc(h2o.performance(md30, dx_test))




#Try XGBoost by using xgboost packagage within R
library(readr)
library(xgboost)
library(ROCR)


d <- read_csv("test.csv")

set.seed(123)
N <- nrow(d)
idx <- sample(1:N, 0.6*N)
d_train <- d[idx,]
d_test <- d[-idx,]


X <- Matrix::sparse.model.matrix(y ~ . - 1, data = d)
X_train <- X[idx,]
X_test <- X[-idx,]

dxgb_train <- xgb.DMatrix(data = X_train, label = ifelse(d_train$y=='Y',1,0))


system.time({
  n_proc <- parallel::detectCores()
  md <- xgb.train(data = dxgb_train, nthread = n_proc, objective = "binary:logistic", 
                  nround = 300, max_depth = 20, eta = 0.1)
})


phat <- predict(md, newdata = X_test)

rocr_pred <- prediction(phat, d_test$y)
performance(rocr_pred, "auc")@y.values[[1]]


#HW3 - End


###No need

###2. Method - SVM
library(e1071)
library(SparseM)
set.seed(123)

d <- read_csv("test.csv")

set.seed(123)
N <- 5000   
idx <- sample(1:N, 0.6*N)
d_train <- d[idx,]
d_test <- d[setdiff(1:N,idx),]


X <- Matrix::sparse.model.matrix(y ~ . - 1, data = d)
X_train <- X[idx,]
X_test <- X[setdiff(1:N,idx),]

system.time({
  md <- svm(x = X_train, y = as.factor(d_train$y), 
            kernel = "radial", gamma = 0.1, cost = 1,
            probability = TRUE)
})


phat <- attr(predict(md, newdata = X_test, probability = TRUE), "probabilities")[,"Y"]

rocr_pred <- prediction(phat, d_test$y)
performance(rocr_pred, "auc")@y.values[[1]]

###2. Method - Neural Network