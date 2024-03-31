import pandas as pd

corn_price_data = "corn-prices-historical-chart-data.csv"

def corn_data_pd(file): # gives date and corn prices for 1959-07-01 to 2024-03-29 then changes the data from daily values to monthly values
    file = pd.read_csv(file)
    file['month'] = pd.to_datetime(file['DATE']).dt.month 
    file = file[file['DATE'] <='2017-02-28' ]
    file.set_index('DATE', inplace=True)
    file.index = pd.to_datetime(file.index)
    file = file.resample('MS').mean()
    
    return file


def corn_predict(file,date):
    #date = input("Enter a date in YYYY-MM-DD format ")
    file = pd.read_csv(file)
    file['month'] = pd.to_datetime(file['DATE']).dt.month
    file2 = file[file["DATE"] >= date]
    file3 = file2[file2['DATE'] <= date]
    file3.set_index('DATE', inplace=True)
    file3.index = pd.to_datetime(file3.index)
    file3 = file3.resample('MS').mean()
  
    return file3
#print(corn_data_pd(corn_price_data))
#print(corn_predict(corn_price_data))




