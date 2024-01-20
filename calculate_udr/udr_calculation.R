args <- commandArgs(trailingOnly = TRUE)

# Default values for the parameters
sup_signal <- "brand"

for (arg in args) {
  keyval <- strsplit(arg, "=")[[1]]
  if (length(keyval) == 2) {
    key <- keyval[1]
    value <- keyval[2]
    
    if (key == "--sup_signal") {
      sup_signal <- value
    }
  }
}

print(paste("Sup_signal:", sup_signal))

library(dplyr)
library(stringr)
library(ggplot2)
library(ggcorrplot)
library(cowplot)
library(stargazer)
library(caret)
library(expm)

delete_columns <- function(df) {
for(i in ncol(df):1)
{
  mean <- mean(df[,i])
  sd <- sd(df[,i])
  if(mean==0 & sd==0)
  {
    df[,i] <- NULL
  }
}
  return(df)
}

calculate_udr <- function(z_i,z_j) {
  df <- matrix(NA,ncol(z_j),ncol(z_i))
  for(i in 1:ncol(z_i))
  {
    for(j in 1:ncol(z_j))
    {
      df[j,i] <- cor(z_i[,i], z_j[,j],method="spearman")
    }
  }
  df <- abs(df)
#  df <- sqrtm(t(df) %*% df)
  udr_row <- 0
  for(i in 1:nrow(df))
  {
    udr_row <- udr_row + (max(df[i,]) * max(df[i,]))/sum(df[i,])
  }
  udr_col <- 0
  for(j in 1:ncol(df))
  {
    udr_col <- udr_col + (max(df[,j]) * max(df[,j]))/sum(df[,j])
  }
  udr_score <- (udr_row+udr_col)/(nrow(df)+ncol(df))
  # print(paste("UDR Score", udr_score))
  return(udr_score)
}

udr_inner <- function(seed,m) {
  focal_df <- read.csv(paste0(sup_signal,"_s",seed,"_mean_params_test2.csv"))
  focal_df$file_name <- NULL
  focal_df <- delete_columns(focal_df)
  udr_inner = 0
  for(i in 1:10)
  {
    if(i!=seed)
    {
      comparison_df <- read.csv(paste0(sup_signal,"_s",i,"_mean_params_test2.csv"))
      comparison_df$file_name <- NULL
      comparison_df <- delete_columns(comparison_df)
      udr_inner <- udr_inner + calculate_udr(focal_df,comparison_df)
      }
  }
  # print(paste("Mean UDR Inner",udr_inner/9))
  return(udr_inner/9)
}

udr_outer <- function()
{
  udr_outer <- 0
  for(i in 1:10)
  {
    udr_outer <- udr_outer + udr_inner(i,m)
  }
  # print(paste("Mean UDR Outer",udr_outer/10))
  return((udr_outer)/10)
}

print(udr_outer())

