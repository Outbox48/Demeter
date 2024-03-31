def import_nec():
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
    
    return train_test_split, LinearRegression, mean_absolute_error, mean_squared_error, corn_data_pd,corn_predict, weather_combined, weather_predict_comb, RandomForestRegressor, MinMaxScaler, StandardScaler,GridSearchCV

def date_input():
    date = input('Enter a date in YYYY-MM-DD format ')
    return date


def data_pull_weather():
    import pandas as pd
    from weather_data import weather_combined
    weather_data = weather_combined('DesMoine_weather_data.csv','Grandcentral_NE.csv', 'Tulsa_airport.csv',"Witchita.csv")
    return weather_data
def data_pull_corn():
    from cornhistprice import corn_data_pd 
    corn_data = corn_data_pd("corn-prices-historical-chart-data.csv")
    return corn_data
def data_pull_weather_predict():
    from weather_data import weather_predict_comb
    date = date_input()
    predict_weather_data = weather_predict_comb(date)
    return predict_weather_data
def data_pull_corn_predict():
    from cornhistprice import corn_predict
    date = date_input()
    corn_predict_data = corn_predict("corn-prices-historical-chart-data.csv",date)
    return corn_predict_data

def data_comb_function():
    weather_data = data_pull_weather()
    
    corn_data = data_pull_corn()
    data_comb = weather_data.join(corn_data)
    return data_comb

def data_comb_pred():
    weather_predict = data_pull_weather_predict()
    
    corn_predict = data_pull_corn_predict()
    data_comb_predict = weather_predict.join(corn_predict)
    return data_comb_predict




def scaler_function():
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    scaler = StandardScaler()
    data_comb = data_comb_function()
    features = scaler.fit_transform(data_comb) # Drop price column from features
    return features

#features = data.drop('DATE')
def model_creation():
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import GridSearchCV
    features = scaler_function()
    data_comb = data_comb_function()
    data_comb_predict = data_comb_pred() 
    target = data_comb['VALUE']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


    model = RandomForestRegressor(n_estimators=100, random_state=42)  # Set the parameters (EXPERIMENT)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [3, 5, 8],  # Maximum depth of individual trees
    'min_samples_split': [2, 5, 10], # Minimum samples required to split a node
    'max_features' : ['auto', 'sqrt'],
}



# predicted_price = model.predict(data_predict)
# print(f"Predicted price: {predicted_price[0]}")



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
    #print("Best Hyperparameters:", best_params)

    # Use the best model for prediction on the test set
    y_pred = best_model.predict(X_test)

    # Evaluate the model's performance on the test set (e.g., calculate MSE, MAE)

    #def model_predict(data_predict):
    # Use the best model for prediction on new data (assuming 'predict_weather_data' is new data)
    predicted_price = best_model.predict(data_comb_predict)
    #print(f"Predicted price: {predicted_price[0]}")
    return predicted_price