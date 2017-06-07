###1. Setup
#Remove Objects
rm(list=ls())

#Clear Memory
gc(reset=TRUE)

#Set Working Directory
setwd("C:/Users/Eve/Dropbox/UCLA Files/Courses/418 Tools of Data Science/STATS 418 - HW4")

#Load packages
library(readr)
library(ROCR)
library(ggplot2)
library(dplyr)
#CSV
data <- read.csv("test.csv")
summary(data)

#Job vs y
subscribe <- data[which(data$y == "Y"),]
nrow(data)
nrow(subscribe)

data_job = subscribe %>%
  group_by(job) %>%
  summarise(n = n())

ggplot(data=data_job, aes(x=job, y=n))+
  geom_bar(stat = "identity", aes(fill = job))+ 
  theme_bw()+ coord_flip() + 
  xlab("Different Types of Jobs") + ylab("Numbers of Subscriptions")

colnames(data)

attach(mtcars)
par(mfrow=c(1,2))
subscribe <- data[which(data$y == "Y"),]
nrow(data)
nrow(subscribe)

data_job = subscribe %>%
  group_by(job) %>%
  summarise(n = n())

ggplot(data=data_job, aes(x=job, y=n))+
  geom_bar(stat = "identity", aes(fill = job))+ 
  theme_bw()+ coord_flip() + 
  xlab("Different Types of Jobs") + ylab("Numbers of Subscriptions")

data_previous = subscribe %>%
  group_by(previous) %>%
  summarise(n = n())

