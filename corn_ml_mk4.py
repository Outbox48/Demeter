import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from cornhistprice import corn_data_pd 
from cornhistprice import corn_predict
from weather_data import weather_combined
from weather_data import weather_predict_comb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV

date = input('Enter a date in YYYY-MM-DD format ')
corn_data = corn_data_pd("corn-prices-historical-chart-data.csv")

corn_predict_data = corn_predict("corn-prices-historical-chart-data.csv",date)
weather_data = weather_combined('DesMoine_weather_data.csv','Grandcentral_NE.csv', 'Tulsa_airport.csv',"Witchita.csv")

predict_weather_data = weather_predict_comb(date)

#attempting here to try random forest I am joining the data together

data = weather_data.join(corn_data)
data = data.interpolate('linear')
#data.columns = data.columns.astype(str)
# print(data)

#print(data)
# print(corn_predict_data)
# print(predict_weather_data)

data_predict = predict_weather_data.join(corn_predict_data)
#data_predict.columns = data_predict.columns.astype(str)
data_predict  = data_predict.interpolate('linear')
print(data_predict)
#print(data_predict)
#print(corn_data)
#print(weather_data)
#scaler = StandardScaler()

scaler = StandardScaler()
features = scaler.fit_transform(data) # Drop price column from features

#features = data.drop('DATE')

target = data['VALUE']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)  # Set the parameters (EXPERIMENT)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")




# predicted_price = model.predict(data_predict)
# print(f"Predicted price: {predicted_price[0]}")

param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [3, 5, 8],  # Maximum depth of individual trees
    'min_samples_split': [2, 5, 10], # Minimum samples required to split a node
    'max_features' : ['auto', 'sqrt'],
}

# Create a Random Forest model object
model = RandomForestRegressor(random_state=42, max_features='auto')

# Create a GridSearchCV object
grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best model and its parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Print the best parameters
print("Best Hyperparameters:", best_params)

# Use the best model for prediction on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model's performance on the test set (e.g., calculate MSE, MAE)

def model_predict(data_predict):
# Use the best model for prediction on new data (assuming 'predict_weather_data' is new data)
    predicted_price = best_model.predict(data_predict)
    print(f"Predicted price: {predicted_price[0]}")