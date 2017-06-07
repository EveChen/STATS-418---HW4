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


#3. Method - Ensemble(rf+GBM+NN)
library(h2o)
h2o.init(nthreads=-1)

dx <- h2o.importFile("test.csv")
dx_split <- h2o.splitFrame(dx, ratios = 0.7, seed = 123)
dx_train <- dx_split[[1]]
dx_test <- dx_split[[2]]
Xnames <- setdiff(names(dx_train),"y")


#Random Forest
system.time({
  md_rf <- h2o.randomForest(x = Xnames, y = "y", training_frame = dx_train, 
                          ntrees = 300,
                          seed = 123,
                          nfolds = 5, fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE)
})

#GBM
system.time({
  md_gbm <- h2o.gbm(x = Xnames, y = "y", training_frame = dx_train, distribution = "bernoulli", 
                 ntrees = 200, max_depth = 10, learn_rate = 0.1, 
                 nbins = 100, seed = 123,
                 nfolds = 5, fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE)    
})

#Neural Network - Choose the highest AUC
system.time({
  md_nn <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
                         activation = "Rectifier", hidden = c(100,100), 
                         adaptive_rate = FALSE, rate = 0.01, rate_annealing = 1e-04, 
                         momentum_start = 0.5, momentum_ramp = 1e5, momentum_stable = 0.99,
                         epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0)
  })


#Ensemble = rf + GBM + NN
System.time({
md_ensemble <- h2o.stackedEnsemble(x = Xnames, y = "y", training_frame = dx_train, 
                              base_models = list(md_rf@model_id, md_gbm@model_id, md_nn@model_id))
})

#Check all the AUCs
h2o.auc(h2o.performance(md_rf, dx_test))
h2o.auc(h2o.performance(md_gbm, dx_test))
h2o.auc(h2o.performance(md_nn, dx_test))
h2o.auc(h2o.performance(md_ensemble, dx_test))


auc_ensemble <- h2o.getModel(md_ensemble@model$metalearner$name)@model$coefficients_table
s
auc_ensemble
