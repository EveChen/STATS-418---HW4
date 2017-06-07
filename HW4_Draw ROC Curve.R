setwd("C:/Users/Eve/Dropbox/UCLA Files/Courses/418 Tools of Data Science/STATS 418 - HW4")
dir()

library(h2o)

h2o.init(nthreads=-1)

#Data sets with validation
dx <- h2o.importFile("test.csv")
dx_split <- h2o.splitFrame(dx, ratios = c(0.6,0.2), seed = 123)
dx_train <- dx_split[[1]]
dx_valid <- dx_split[[2]]
dx_test <- dx_split[[3]]
Xnames <- setdiff(names(dx_train),"y")

system.time({
  md_logistic <- h2o.glm(x = Xnames, y = "y", training_frame = dx_train, 
                         family = "binomial", 
                         alpha = 1, lambda = 0,
                         seed = 123,
                         nfolds = 5, fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE)
})


h2o.auc(h2o.performance(md_logistic, dx_test))

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