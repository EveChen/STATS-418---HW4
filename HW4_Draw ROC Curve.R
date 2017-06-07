setwd("C:/Users/Eve/Dropbox/UCLA Files/Courses/418 Tools of Data Science/STATS 418 - HW4")
dir()
####**1. Neural Network**
```{r, include = F}
library(h2o)
```
```{r, include = F}
h2o.no_progress()
```

```{r, include = F}
h2o.init(nthreads=-1)
```

```{r}
#Split data sets: train, validation and test
dx <- h2o.importFile("test.csv")
dx_split <- h2o.splitFrame(dx, ratios = c(0.6,0.2), seed = 123)
dx_train <- dx_split[[1]]
dx_valid <- dx_split[[2]]
dx_test <- dx_split[[3]]
Xnames <- setdiff(names(dx_train),"y")
```

```{r}
#Neural Network Model (the Best, highest AUC)
system.time({
  md_nn <- h2o.deeplearning(x = Xnames, y = "y", training_frame = dx_train, validation_frame = dx_valid,
                            activation = "Rectifier", hidden = c(50,50), 
                            adaptive_rate = FALSE, rate = 0.01, rate_annealing = 1e-04, 
                            momentum_start = 0.5, momentum_ramp = 1e5, momentum_stable = 0.99,
                            epochs = 100, stopping_rounds = 2, stopping_metric = "AUC", stopping_tolerance = 0) 
})
```
```{r}
#AUC
h2o.performance(md_nn, dx_test)@metrics$AUC
```

#Q: Cannot draw ROC curve
#Try ROCR to make ROC curve with True/False Positive

#Try1 - ROCR
library(ROCR)
roc <- performance(md_nn, measure = "tpr", x.measure = "fpr")


```{r, include = F}
#Try2 - pROC
library(pROC)
```

```{r}
#ROC for Neural Network model
phat_nn <- predict(md_nn, newx = dx_test, type = "response")
ROC_nn = roc(dx_test$y, phat)
plot(ROC_nn)
```


#Try3 - Failed
library(pROC)
glm_link_scores <- predict(md_nn, dx_test, type="link")
pred <- predict(md_nn, dx_test, type = "response")

simple_roc <- function(labels, scores){
  labels <- labels[order(scores, decreasing=TRUE)]
  data.frame(TPR=cumsum(labels)/sum(labels), FPR=cumsum(!labels)/sum(!labels), labels)
}

glm_simple_roc <- simple_roc(dx_test$y=="Y", glm_link_scores)


plot(roc(dx_test$y, pred, direction="<"),
     col="yellow", lwd=3, main="The turtle finds its way")
## 