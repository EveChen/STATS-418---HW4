
library(h2o)

h2o.init(nthreads=-1)


dx <- h2o.importFile("test.csv")

dx_split <- h2o.splitFrame(dx, ratios = c(0.6,0.2), seed = 123)
dx_train <- dx_split[[1]]
dx_valid <- dx_split[[2]]
dx_test <- dx_split[[3]]


Xnames <- names(dx_train)[which(names(dx_train)!="y")]

#Only test for NN which was built within Ensemble example
###Data set - New
den <- h2o.importFile("test.csv")
den_split <- h2o.splitFrame(den, ratios = 0.7, seed = 123)
den_train <- den_split[[1]]
den_test <- den_split[[2]]
Xnamesen <- setdiff(names(den_train),"y")

###New model - built within Ensemble example
system.time({
  md_nn <- h2o.deeplearning(x = Xnamesen, y = "y", training_frame = den_train,
                            activation = "Rectifier", hidden = c(100,100), 
                            adaptive_rate = FALSE, rate = 0.01, 
                            rate_annealing = 1e-04,momentum_start = 0.5, 
                            momentum_ramp = 1e5, momentum_stable = 0.99, 
                            epochs = 100, seed = 123, nfolds = 5, 
                            fold_assignment = "Modulo")
})
h2o.performance(md_nn, den_test)@metrics$AUC


#Back to the original Neural Network models

system.time({
  md <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid, epochs = 100, stopping_rounds = 2, hidden = c(100,100), stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(50,50,50,50), input_dropout_ratio = 0.2,
            epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(50,50,50,50), 
            epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(20,20),
            epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(20),
            epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(5),
            epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(1),
            epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(100,100), l1 = 1e-5, l2 = 1e-5, 
            epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
            activation = "RectifierWithDropout", hidden = c(100,100,100,100), hidden_dropout_ratios=c(0.2,0.1,0.1,0),
            epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(100,100), 
            rho = 0.95, epsilon = 1e-06,  ## default:  rho = 0.99, epsilon = 1e-08
            epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(100,100), 
            rho = 0.999, epsilon = 1e-08,  ## default:  rho = 0.99, epsilon = 1e-08
            epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(100,100), 
            rho = 0.9999, epsilon = 1e-08,  ## default:  rho = 0.99, epsilon = 1e-08
            epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(100,100), 
            rho = 0.999, epsilon = 1e-06,  ## default:  rho = 0.99, epsilon = 1e-08
            epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(100,100), 
            rho = 0.999, epsilon = 1e-09,  ## default:  rho = 0.99, epsilon = 1e-08
            epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(100,100), 
            adaptive_rate = FALSE, ## default: rate = 0.005, rate_decay = 1, momentum_stable = 0,
            epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(100,100), 
            adaptive_rate = FALSE, rate = 0.001, momentum_start = 0.5, momentum_ramp = 1e5, momentum_stable = 0.99,
            epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(100,100), 
            adaptive_rate = FALSE, rate = 0.01, momentum_start = 0.5, momentum_ramp = 1e5, momentum_stable = 0.99,
            epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(100,100), 
            adaptive_rate = FALSE, rate = 0.01, rate_annealing = 1e-05, 
            momentum_start = 0.5, momentum_ramp = 1e5, momentum_stable = 0.99,
            epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(100,100), 
            adaptive_rate = FALSE, rate = 0.01, rate_annealing = 1e-04, 
            momentum_start = 0.5, momentum_ramp = 1e5, momentum_stable = 0.99,
            epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(100,100), 
            adaptive_rate = FALSE, rate = 0.01, rate_annealing = 1e-05, 
            momentum_start = 0.5, momentum_ramp = 1e5, momentum_stable = 0.9,
            epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(100,100), 
            adaptive_rate = FALSE, rate = 0.01, rate_annealing = 1e-05, 
            momentum_start = 0.5, momentum_ramp = 1e4, momentum_stable = 0.9,
            epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC




