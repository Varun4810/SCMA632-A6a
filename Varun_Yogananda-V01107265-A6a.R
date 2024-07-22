#install.packages(c("quantmod", "xts", "forecast", "ggplot2", "tseries", "prophet"))

library(quantmod)
library(xts)
library(forecast)
library(ggplot2)
library(tseries)
library(prophet)

# Get the data for Wipro Limited
data=read.csv("D:/Bootcamp VCU datasets/WIT (1).csv")
head(data)

# Check for missing values
sum(is.na(data))

# Convert the Date column to Date type
data$Date= as.Date(data$Date, format="%Y-%m-%d")

# Convert data to xts object
data_xts= xts(data$Adj.Close, order.by=data$Date)

# Plot the Adjusted Close Price
plot(data_xts, main="Adjusted Close Price", xlab="Date", ylab="Adjusted Close Price", col="blue", type="l")

# the time series is at a monthly frequency
data_monthly=to.monthly(data_xts, indexAt='lastof', OHLC=FALSE)

# Convert to a ts object
data_ts= ts(data_monthly, frequency=12)

# Decompose the time series
decomp =decompose(data_ts, type="multiplicative")

# Plot the decomposed components
plot(decomp)


# Split the data into training and test sets (80-20 split)
train_size <- floor(0.8 * length(data_ts))
train_data <- window(data_ts, end = c(time(data_ts)[train_size]))
test_data <- window(data_ts, start = c(time(data_ts)[train_size + 1]))

# Apply Holt-Winters exponential smoothing
holt_winters_model <- HoltWinters(train_data)
holt_winters_forecast <- forecast(holt_winters_model, h = length(test_data))

# Compute RMSE
rmse <- sqrt(mean((test_data - holt_winters_forecast$mean)^2))
print(paste("RMSE:", rmse))

# Compute MAE
mae = mean(abs(test_data- holt_winters_forecast$mean))
print(paste("MAE:", mae))

# Compute MAPE
mape = mean(abs((test_data-holt_winters_forecast$mean) / test_data)) * 100
print(paste("MAPE:", mape))

# Compute the R-squared value
ss_res = sum((actual_values - forecast_values)^2)
ss_tot= sum((actual_values - mean(actual_values))^2)
r_squared= 1 - (ss_res / ss_tot)
print(paste("R-squared:", r_squared))


# Split the data into training and test sets (80-20 split)
train_size = floor(0.8 * length(data))
train= data[1:train_size]
test= data[(train_size + 1):length(data)]

#Univariate Forecasting with Holt-Winters Model

train_length =length(data_ts) - 12
test_data = window(data_ts, start=c(2024, 6))
train_data =window(data_ts, end=c(2024, 7)) 

# Fit the Holt-Winters model
holt_winters_model=HoltWinters(train_data, seasonal="multiplicative")

# Forecast for the next 12 months
holt_winters_forecast = forecast(holt_winters_model, h=12)

# Plot the forecast
plot(holt_winters_forecast, main="Holt-Winters Forecast", xlab="Date", ylab="Adjusted Close Price")
lines(train_data, col="blue") 



# Fit ARIMA model
arima_model <- auto.arima(train_data)
arima_forecast <- forecast(arima_model, h = length(test_data))

# Plot ARIMA forecast
plot(arima_forecast, main="ARIMA Forecast", xlab="Date", ylab="Adjusted Close Price")
lines(train_data, col="red")



# Print the model summary
summary(arima_model)

# Forecast for the next period (e.g., next 12 months)
forecast_values = forecast(arima_model, h=12)

# Print the forecast
print(forecast_values)

# Plot the forecast
plot(forecast_values)

# *************************
#2. Multivariate Forecasting - Machine Learning Models
  
#install.packages("keras")
library(keras)
#install_keras()


library(dplyr)
library(tidyr)
library(lubridate)
library(keras)

data=read.csv("D:/Bootcamp VCU datasets/WIT (1).csv")
head(data)

# Select features and target
features= data %>% select(-Adj.Close)
target = data %>% select(Adj.Close)

str(features)
# Select numeric columns for features
numeric_features <- features %>% select_if(is.numeric)


# Select target
target <- data %>% select(Adj.Close)

# Normalize features and target
scaled_features =as.data.frame(scale(numeric_features))
scaled_target = as.data.frame(scale(target))

# Create scaled data frame
scaled_df =cbind(scaled_features, Adj.Close = scaled_target)


sequence_length <- 30  # Define the sequence length

create_sequences <- function(data, target_col, sequence_length) {
  num_samples <- nrow(data) - sequence_length
  num_features <- ncol(data)
  
  sequences <- array(NA, dim = c(num_samples, sequence_length, num_features))
  labels <- numeric(num_samples)
  
  for (i in 1:num_samples) {
    sequences[i,,] <- as.matrix(data[i:(i + sequence_length - 1),])
    labels[i] <- data[i + sequence_length, target_col]
  }
  return(list(sequences, labels))
}

# Convert DataFrame to matrix
data_matrix <- as.matrix(scaled_df)

# Define the target column index
target_col <- which(colnames(scaled_df) == "Adj.Close")

# Create sequences
sequences_and_labels <- create_sequences(data_matrix, target_col, sequence_length)
X <- sequences_and_labels[[1]]
y <- sequences_and_labels[[2]]

train_size <- floor(0.8 * nrow(X))
X_train <- X[1:train_size,,]
y_train <- y[1:train_size]
X_test <- X[(train_size + 1):nrow(X),,]
y_test <- y[(train_size + 1):length(y)]


# Build the LSTM model
model <- keras_model_sequential() %>%
  layer_lstm(units = 50, return_sequences = TRUE, input_shape = c(sequence_length, ncol(X_train[1,,]))) %>%
  layer_dropout(rate = 0.2) %>%
  layer_lstm(units = 50, return_sequences = FALSE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1)

model %>% compile(
  optimizer = 'adam',
  loss = 'mean_squared_error'
)


# Train the model
history <- model %>% fit(
  X_train, y_train,
  epochs = 20,
  batch_size = 32,
  validation_data = list(X_test, y_test),
  shuffle = FALSE
)

attributes(scaled_target)

print("First 10 values of y_test:")
print(head(y_test, 10))

print("First 10 values of y_pred:")
print(head(y_pred, 10))

print("Attributes of scaled_target:")
print(attributes(scaled_target))

print("Center:")
print(center)

print("Scale:")
print(scale)

y_test <- as.numeric(y_test)
y_pred <- as.numeric(y_pred)

# Assuming mean and sd used for scaling
mean_target <- mean(y_test, na.rm = TRUE)
sd_target <- sd(y_test, na.rm = TRUE)

# Manual scaling back
y_test_scaled <- (y_test * sd_target) + mean_target
y_pred_scaled <- (y_pred * sd_target) + mean_target


# Compute RMSE
rmse <- sqrt(mean((y_test_scaled - y_pred_scaled)^2, na.rm = TRUE))
print(paste("RMSE:", rmse))

# Compute MAE
mae <- mean(abs(y_test_scaled - y_pred_scaled), na.rm = TRUE)
print(paste("MAE:", mae))

# Compute MAPE
mape <- mean(abs((y_test_scaled - y_pred_scaled) / y_test_scaled), na.rm = TRUE) * 100
print(paste("MAPE:", mape))

# Compute R-squared
ss_res <- sum((y_test_scaled - y_pred_scaled)^2, na.rm = TRUE)
ss_tot <- sum((y_test_scaled - mean(y_test_scaled, na.rm = TRUE))^2, na.rm = TRUE)
r2 <- 1 - (ss_res / ss_tot)
print(paste("R-squared:", r2))

length(y_test)
length(y_pred)

# Truncate y_pred to match the length of y_test
y_pred <- y_pred[1:length(y_test)]


# Create a data frame for plotting
df_plot <- data.frame(
  Time = 1:length(y_test),
  True_Values = y_test,
  Predictions = y_pred
)

library(ggplot2)

# Plot both True Values and Predictions
ggplot(df_plot, aes(x = Time)) +
  geom_line(aes(y = True_Values, color = "True Values")) +
  geom_line(aes(y = Predictions, color = "Predictions")) +
  labs(title = "LSTM: Predictions vs True Values",
       x = "Time",
       y = "Value") +
  scale_color_manual(values = c("True Values" = "blue", "Predictions" = "red")) +
  theme_minimal()


#install.packages("randomForest")

library(randomForest)
library(rpart)
library(caret)
library(lattice)

# Flatten X for Decision Tree
num_samples <- nrow(X)
seq_length <- dim(X)[2]
num_features <- dim(X)[3]
X_flattened <- matrix(X, nrow = num_samples, ncol = seq_length * num_features)

# Split data into train and test sets
set.seed(42)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X_flattened[trainIndex, ]
X_test <- X_flattened[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

# Train Decision Tree model
dt_model <- rpart(y_train ~ ., data = data.frame(y_train = y_train, X_train))
y_pred_dt <- predict(dt_model, newdata = data.frame(X_test))

# Evaluate the Decision Tree model
mse_dt <- mean((y_test - y_pred_dt)^2)
cat("MSE (Decision Tree):", mse_dt, "\n")

# Compute RMSE
rmse_dt <- sqrt(mse_dt)
cat("RMSE (Decision Tree):", rmse_dt, "\n")

# Compute MAE
mae_dt <- mean(abs(y_test - y_pred_dt))
cat("MAE (Decision Tree):", mae_dt, "\n")

# Compute MAPE
mape_dt <- mean(abs((y_test - y_pred_dt) / y_test)) * 100
cat("MAPE (Decision Tree):", mape_dt, "\n")

# Compute R-squared
r2_dt <- 1 - (sum((y_test - y_pred_dt)^2) / sum((y_test - mean(y_test))^2))
cat("R-squared (Decision Tree):", r2_dt, "\n")

# Train Random Forest model
rf_model <- randomForest(x = X_train, y = y_train, ntree = 100)
y_pred_rf <- predict(rf_model, newdata = X_test)

# Evaluate the Random Forest model
mse_rf <- mean((y_test - y_pred_rf)^2)
cat("MSE (Random Forest):", mse_rf, "\n")

# Compute RMSE
rmse_rf <- sqrt(mse_rf)
cat("RMSE (Random Forest):", rmse_rf, "\n")

# Compute MAE
mae_rf <- mean(abs(y_test - y_pred_rf))
cat("MAE (Random Forest):", mae_rf, "\n")

# Compute MAPE
mape_rf <- mean(abs((y_test - y_pred_rf) / y_test)) * 100
cat("MAPE (Random Forest):", mape_rf, "\n")

# Compute R-squared
r2_rf <- 1 - (sum((y_test - y_pred_rf)^2) / sum((y_test - mean(y_test))^2))
cat("R-squared (Random Forest):", r2_rf, "\n")


# Plot the predictions vs true values for Decision Tree
df_plot_dt <- data.frame(
  Time = 1:length(y_test),
  True_Values = y_test,
  Predictions = y_pred_dt
)

ggplot(df_plot_dt, aes(x = Time)) +
  geom_line(aes(y = True_Values, color = 'True Values')) +
  geom_line(aes(y = Predictions, color = 'Decision Tree Predictions')) +
  labs(title = 'Decision Tree: Predictions vs True Values', y = 'Close Price') +
  theme_minimal()

# Plot the predictions vs true values for Random Forest
df_plot_rf <- data.frame(
  Time = 1:length(y_test),
  True_Values = y_test,
  Predictions = y_pred_rf
)

ggplot(df_plot_rf, aes(x = Time)) +
  geom_line(aes(y = True_Values, color = 'True Values')) +
  geom_line(aes(y = Predictions, color = 'Random Forest Predictions')) +
  labs(title = 'Random Forest: Predictions vs True Values', y = 'Close Price') +
  theme_minimal()



# Create a data frame for plotting
df_plot_both <- data.frame(
  Time = 1:length(y_test),
  True_Values = y_test,
  Decision_Tree_Predictions = y_pred_dt,
  Random_Forest_Predictions = y_pred_rf
)

# Plot both Decision Tree and Random Forest predictions together
ggplot(df_plot_both, aes(x = Time)) +
  geom_line(aes(y = True_Values, color = "True Values"), size = 1) +
  geom_line(aes(y = Decision_Tree_Predictions, color = "Decision Tree Predictions"), linetype = "dashed") +
  geom_line(aes(y = Random_Forest_Predictions, color = "Random Forest Predictions"), linetype = "dotted") +
  labs(title = "True Values vs Predictions",
       x = "Time",
       y = "Values") +
  scale_color_manual(values = c("True Values" = "black", 
                                "Decision Tree Predictions" = "blue", 
                                "Random Forest Predictions" = "red")) +
  theme_minimal()

