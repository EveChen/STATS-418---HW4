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

#CSV
dat <- read.csv("test.csv")
summary(dat)
