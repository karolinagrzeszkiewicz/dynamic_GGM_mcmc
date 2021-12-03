library(Matrix)
library(rv)
library(MASS)
library(utils)

p = 10

P1 = diag(p)

for (i in 1:(p-1)) {
  P1[i, i+1] = 0.5
  P1[i+1,i] = 0.5
}

for (i in 1:(p-2)) {
  P1[i, i+2] = 0.4
  P1[i+2,i] = 0.4
}

getNonZeroElem <- function(M) {
  lsi <- list()
  lsj <- list()
  for (i in 1:(p-1)) {
    for (j in (i+1):p) {
      if (M[i,j] != 0.0) {
        lsi <- append(lsi, i)
        lsj <- append(lsj, j)
      }
    }
  }
  return(cbind(lsi, lsj))
}

removeEdgeRand <- function(M) {
  indices = getNonZeroElem(M)
  l = length(indices)/2
  pick <- sample(1:l, 5, replace=F)
  for (idx in pick) {
    i <- as.integer(indices[idx,1])
    j <- as.integer(indices[idx,2])
    M[i,j] = 0
    M[j,i] = 0
  }
  return(M)
}

getZeroElem <- function(M) {
  lsi2 <- list()
  lsj2 <- list()
  for (i in 1:(p-1)) {
    for (j in (i+1):p) {
      if (M[i,j] == 0.0) {
        lsi2 <- append(lsi2, i)
        lsj2 <- append(lsj2, j)
      }
    }
  }
  return(cbind(lsi2, lsj2))
}

addEdgeRand <- function(M) {
  indices = getZeroElem(M)
  l = length(indices)/2
  pick <- sample(1:l, 5, replace=F)
  for (idx in pick) {
    i <- as.integer(indices[idx,1])
    j <- as.integer(indices[idx,2])
    M[i,j] = 0.5
    M[j,i] = 0.5
  }
  return(M)
}
  

P2 = P1

P2 = removeEdgeRand(P2)
P2 = addEdgeRand(P2)

P3 = P2

P3 = removeEdgeRand(P3)
P3 = addEdgeRand(P3)

P4 = P3

P4 = removeEdgeRand(P4)
P4 = addEdgeRand(P4)

P1 = nearPD(P1)$mat
P2 = nearPD(P2)$mat
P3 = nearPD(P3)$mat
P4 = nearPD(P4)$mat

#dpoMatrix class

#GENERATE DATA 

# NO CHANGEPOINTS T = 50, DATA FROM P2 Y_S0_P2
T = 50

mean = numeric(p)

C2 = solve(P2) #covariance matrix

#for (i in 1:T)
#{y[i,1:p] = rvnorm(n = 1, mean = mean, precision = P2)}

Y_S0_P2 = mvrnorm(T, mean, C2)

dirname <- dirname(rstudioapi::getSourceEditorContext()$path)

write.csv(Y_S0_P2, file = paste(dirname, "Y_T_50_S0_P2_0.csv", sep = "/"))

#repeat 

Y_S0_P2_1 = mvrnorm(T, mean, C2)

dirname <- dirname(rstudioapi::getSourceEditorContext()$path)

write.csv(Y_S0_P2_1, file = paste(dirname, "Y_T_50_S0_P2_1.csv", sep = "/"))


Y_S0_P2_2 = mvrnorm(T, mean, C2)

dirname <- dirname(rstudioapi::getSourceEditorContext()$path)

write.csv(Y_S0_P2_2, file = paste(dirname, "Y_T_50_S0_P2_2.csv", sep = "/"))


Y_S0_P2_3 = mvrnorm(T, mean, C2)

dirname <- dirname(rstudioapi::getSourceEditorContext()$path)

write.csv(Y_S0_P2_3, file = paste(dirname, "Y_T_50_S0_P2_3.csv", sep = "/"))


Y_S0_P2_4 = mvrnorm(T, mean, C2)

dirname <- dirname(rstudioapi::getSourceEditorContext()$path)

write.csv(Y_S0_P2_4, file = paste(dirname, "Y_T_50_S0_P2_4.csv", sep = "/"))


#1 CHANGEPOINT

C3 = solve(P3)

yC2 = mvrnorm(T, mean, C2)

yC3 = mvrnorm(T, mean, C3)

Y_S1_P2_P3_0 = rbind(yC2,yC3)

write.csv(Y_S1_P2_P3_0, file = paste(dirname, "Y_S1_P2_P3_0.csv", sep = "/"))

#repeat

yC2 = mvrnorm(T, mean, C2)

yC3 = mvrnorm(T, mean, C3)

Y_S1_P2_P3_1 = rbind(yC2,yC3)

write.csv(Y_S1_P2_P3_1, file = paste(dirname, "Y_S1_P2_P3_1.csv", sep = "/"))


#repeat

yC2 = mvrnorm(T, mean, C2)

yC3 = mvrnorm(T, mean, C3)

Y_S1_P2_P3_2 = rbind(yC2,yC3)

write.csv(Y_S1_P2_P3_2, file = paste(dirname, "Y_S1_P2_P3_2.csv", sep = "/"))

#repeat

yC2 = mvrnorm(T, mean, C2)

yC3 = mvrnorm(T, mean, C3)

Y_S1_P2_P3_3 = rbind(yC2,yC3)

write.csv(Y_S1_P2_P3_3, file = paste(dirname, "Y_S1_P2_P3_3.csv", sep = "/"))

#repeat

yC2 = mvrnorm(T, mean, C2)

yC3 = mvrnorm(T, mean, C3)

Y_S1_P2_P3_4 = rbind(yC2,yC3)

write.csv(Y_S1_P2_P3_4, file = paste(dirname, "Y_S1_P2_P3_4.csv", sep = "/"))

#2 changepoints

C1 = solve(P1)

yC1 = mvrnorm(T, mean, C1)

yC2 = mvrnorm(T, mean, C2)

yC3 = mvrnorm(T, mean, C3)

Y_S2_P1_P2_P3_0 = rbind(yC1,yC2,yC3)

write.csv(Y_S2_P1_P2_P3_0, file = paste(dirname, "Y_S2_P1_P2_P3_0.csv", sep = "/"))

#repeat

yC1 = mvrnorm(T, mean, C1)

yC2 = mvrnorm(T, mean, C2)

yC3 = mvrnorm(T, mean, C3)

Y_S2_P1_P2_P3_1 = rbind(yC1,yC2,yC3)

write.csv(Y_S2_P1_P2_P3_1, file = paste(dirname, "Y_S2_P1_P2_P3_1.csv", sep = "/"))

#repeat

yC1 = mvrnorm(T, mean, C1)

yC2 = mvrnorm(T, mean, C2)

yC3 = mvrnorm(T, mean, C3)

Y_S2_P1_P2_P3_2 = rbind(yC1,yC2,yC3)

write.csv(Y_S2_P1_P2_P3_2, file = paste(dirname, "Y_S2_P1_P2_P3_2.csv", sep = "/"))

#repeat

yC1 = mvrnorm(T, mean, C1)

yC2 = mvrnorm(T, mean, C2)

yC3 = mvrnorm(T, mean, C3)

Y_S2_P1_P2_P3_3 = rbind(yC1,yC2,yC3)

write.csv(Y_S2_P1_P2_P3_3, file = paste(dirname, "Y_S2_P1_P2_P3_3.csv", sep = "/"))

#repeat

yC1 = mvrnorm(T, mean, C1)

yC2 = mvrnorm(T, mean, C2)

yC3 = mvrnorm(T, mean, C3)

Y_S2_P1_P2_P3_4 = rbind(yC1,yC2,yC3)

write.csv(Y_S2_P1_P2_P3_4, file = paste(dirname, "Y_S2_P1_P2_P3_4.csv", sep = "/"))

#2 changepoints

C4 = solve(P4)

yC1 = mvrnorm(T, mean, C1)

yC2 = mvrnorm(T, mean, C2)

yC3 = mvrnorm(T, mean, C3)

yC4 = mvrnorm(T, mean, C4)

Y_S3_P1_P2_P3_P4_0 = rbind(yC1,yC2,yC3,yC4)

write.csv(Y_S3_P1_P2_P3_P4_0, file = paste(dirname, "Y_S3_P1_P2_P3_P4_0.csv", sep = "/"))

#repeat


yC1 = mvrnorm(T, mean, C1)

yC2 = mvrnorm(T, mean, C2)

yC3 = mvrnorm(T, mean, C3)

yC4 = mvrnorm(T, mean, C4)

Y_S3_P1_P2_P3_P4_1 = rbind(yC1,yC2,yC3,yC4)

write.csv(Y_S3_P1_P2_P3_P4_1, file = paste(dirname, "Y_S3_P1_P2_P3_P4_1.csv", sep = "/"))

#repeat 

yC1 = mvrnorm(T, mean, C1)

yC2 = mvrnorm(T, mean, C2)

yC3 = mvrnorm(T, mean, C3)

yC4 = mvrnorm(T, mean, C4)

Y_S3_P1_P2_P3_P4_2 = rbind(yC1,yC2,yC3,yC4)

write.csv(Y_S3_P1_P2_P3_P4_2, file = paste(dirname, "Y_S3_P1_P2_P3_P4_2.csv", sep = "/"))

#repeat 

yC1 = mvrnorm(T, mean, C1)

yC2 = mvrnorm(T, mean, C2)

yC3 = mvrnorm(T, mean, C3)

yC4 = mvrnorm(T, mean, C4)

Y_S3_P1_P2_P3_P4_3 = rbind(yC1,yC2,yC3,yC4)

write.csv(Y_S3_P1_P2_P3_P4_3, file = paste(dirname, "Y_S3_P1_P2_P3_P4_3.csv", sep = "/"))

#repeat 

yC1 = mvrnorm(T, mean, C1)

yC2 = mvrnorm(T, mean, C2)

yC3 = mvrnorm(T, mean, C3)

yC4 = mvrnorm(T, mean, C4)

Y_S3_P1_P2_P3_P4_4 = rbind(yC1,yC2,yC3,yC4)

write.csv(Y_S3_P1_P2_P3_P4_4, file = paste(dirname, "Y_S3_P1_P2_P3_P4_4.csv", sep = "/"))


#save precision matrices
write.csv(matrix(P1@x, nrow = 10, ncol=10), file = paste(dirname, "P1.csv", sep = "/"))
write.csv(matrix(P2@x, nrow = 10, ncol=10), file = paste(dirname, "P2.csv", sep = "/"))
write.csv(matrix(P3@x, nrow = 10, ncol=10), file = paste(dirname, "P3.csv", sep = "/"))
write.csv(matrix(P4@x, nrow = 10, ncol=10), file = paste(dirname, "P4.csv", sep = "/"))

