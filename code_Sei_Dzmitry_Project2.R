library(robustbase)

setwd("D:\\seidm\\leuven\\2\\robust\\project_2")


set.seed(0913246)

traindata <- read.csv("StarGalaxy_train_RobSta.csv", row.names=1)
testdata <- read.csv("StarGalaxy_test_RobSta.csv", row.names=1)


nsamp <- 1000
mytraindata <- traindata[sample(nrow(traindata), nsamp), ]
mytestdata  <- read.csv("StarGalaxy_test_RobSta.csv", row.names=1)


################### Robust Estimation of the Full Model ########################


# Build standard non-robust logistic regression
log_glm = glm(Class ~ ., data = mytraindata, family = binomial)


# Build robust logistic regression
rob_glm = glmrob(Class ~ ., data = mytraindata, family = binomial, 
                 control= glmrobMqle.control(tcc=1.8), method= "Mqle", weights.on.x = 'h')

# model analytics
log_glm_summary = summary(log_glm)
rob_glm_summary = summary(rob_glm)

log_glm_is_significant = log_glm_summary$coefficients[, 4] < 0.05
rob_glm_is_significant = rob_glm_summary$coefficients[, 4] < 0.05

common_summary = cbind(round(log_glm_summary$coefficients[, colnames(log_glm_summary$coefficients)], 5),
                       ifelse(log_glm_is_significant, 'T', 'F'),
                       round(rob_glm_summary$coefficients[, colnames(rob_glm_summary$coefficients)], 5),
                       ifelse(rob_glm_is_significant, 'T', 'F'))

common_summary

# significance inconsistency between the models                       
which(!(log_glm_is_significant == rob_glm_is_significant))


colnames(common_summary) <- c('Non-robust_Estimate', 'Non-robust_Std', 'Non-robust_z_value', 'Non-robust_p', 'Non-robust_Is_significant',
                              'Robust_Estimate', 'Robust_Std', 'Robust_z_value', 'Robust_p', 'Robust_Is_significant')

summarizeRobWeights(rob_glm$w.x)


# common_summary[,'Non-robust_Estimate'] 
common_summary

estimates_fraction = cbind(
  log_glm_summary$coefficients[, 'Estimate'] / rob_glm_summary$coefficients[, 'Estimate'],
  log_glm_summary$coefficients[, 'Std. Error'] / rob_glm_summary$coefficients[, 'Std. Error'])
  
colnames(estimates_fraction) <- c('Non-rob. Estimate/rob Estimate', 'Non-rob. Std. Error/rob. Std. Error')
estimates_fraction

################### Full Model outliers detection ##############################


summary(rob_glm)

# Outliers consistent with Cantoni 2006 approach
which(rob_glm$w.r < 0.5)
length(which(rob_glm$w.r < 0.5))

# Plot the data, with named outliers
plot(rob_glm$w.r, pch = 16, cex = 1.5, ylim = c(0, 1), ylab = "Values", xlab = "Indexes")
abline(h = 0.5, col = "red", lty = 2)
names_below <- which(rob_glm$w.r < 0.5)
text(names_below, rob_glm$w.r[names_below], labels = names_below, pos = 3)


################### Model selection ############################################

summary(rob_glm)
colnames(traindata)

# Use covariates except [MTotF, MAperJ, MTotJ, MCoreJ, EllipJ, AreaN, IR2N, csfN, EllipN]
rob_glm_truncated = glmrob(Class ~ MAperF  + MCoreF + AreaF + IR2F + csfF + EllipF
                   + AreaJ + IR2J + csfJ  
                   + MAperN + MTotN + MCoreN, data = mytraindata, family = binomial, 
                 control= glmrobMqle.control(tcc=1.8), method= "Mqle", weights.on.x = 'h')

anova(rob_glm_truncated, rob_glm, test="QD")



################### Models perfomance on the test data ##########################


log_glm_truncated = glm(Class ~ MAperF  + MCoreF + AreaF + IR2F + csfF + EllipF
                        + AreaJ + IR2J + csfJ  
                        + MAperN + MTotN + MCoreN, 
                        data = mytraindata, family = binomial)



# Load required packages
library(pROC)

# Function to find optimal cutoff point based on AUC
find_optimal_cutoff <- function(true_labels, predicted_probs) {
  roc_obj <- roc(true_labels, predicted_probs)
  coords <- coords(roc_obj, "best")
  return(coords["threshold"])
}

train_labels <- mytraindata$Class
test_labels <- mytestdata$Class

rob_probs_preds_train <- predict(rob_glm, newdata = mytraindata, type = "response")
log_probs_preds_train <- predict(log_glm, newdata = mytraindata, type = "response")
rob_probs_trunc_preds_train <-predict(rob_glm_truncated, newdata = mytraindata, type = "response")
log_probs_trunc_preds_train <- predict(log_glm_truncated, newdata = mytraindata, type = "response")

# Find optimal cutoff points for each model on the training data
optimal_cutoff_rob_glm <- find_optimal_cutoff(train_labels, rob_probs_preds_train)
optimal_cutoff_log_glm <- find_optimal_cutoff(train_labels, log_probs_preds_train)
optimal_cutoff_rob_glm_truncated <- find_optimal_cutoff(train_labels, rob_probs_trunc_preds_train)
optimal_cutoff_log_glm_truncated <- find_optimal_cutoff(train_labels, log_probs_trunc_preds_train)

# Function to apply cutoff and classify probabilities
apply_cutoff <- function(probabilities, cutoff) {
  return(as.numeric(probabilities > cutoff))
}


rob_prob_preds_test <- predict(rob_glm, newdata = mytestdata, type = "response")
log_prob_preds_test <- predict(log_glm, newdata = mytestdata, type = "response")
rob_trunc_prob_preds_test <-predict(rob_glm_truncated, newdata = mytestdata, type = "response")
log_trunc_prob_preds_test <- predict(log_glm_truncated, newdata = mytestdata, type = "response")


# Apply optimal cutoff to test data predictions
test_predictions_rob_glm <- apply_cutoff(rob_prob_preds_test, optimal_cutoff_rob_glm$threshold)
test_predictions_log_glm <- apply_cutoff(log_prob_preds_test, optimal_cutoff_log_glm$threshold)
test_predictions_rob_glm_truncated <- apply_cutoff(rob_trunc_prob_preds_test, optimal_cutoff_rob_glm_truncated$threshold)
test_predictions_log_glm_truncated <- apply_cutoff(log_trunc_prob_preds_test, optimal_cutoff_log_glm_truncated$threshold)

# Evaluate performance on test data
test_auc_rob_glm <- roc(test_labels, test_predictions_rob_glm)$auc
test_auc_log_glm <- roc(test_labels, test_predictions_log_glm)$auc
test_auc_rob_glm_truncated <- roc(test_labels, test_predictions_rob_glm_truncated)$auc
test_auc_log_glm_truncated <- roc(test_labels, test_predictions_log_glm_truncated)$auc

# Print AUC values for each model on test data
cat("AUC on test data for rob_glm:", test_auc_rob_glm, "\n")
cat("AUC on test data for log_glm:", test_auc_log_glm, "\n")
cat("AUC on test data for rob_glm_truncated:", test_auc_rob_glm_truncated, "\n")
cat("AUC on test data for log_glm_truncated:", test_auc_log_glm_truncated, "\n")




