import pandas as pd



weather_data_dm = "DesMoine_weather_data.csv"

  #reading the file into a pandas database
file1 = "DesMoine_weather_data.csv"
file2 = "Tulsa_airport.csv"
file3 = "Tulsa_airport.csv"
file4 = "Grandcentral_NE.csv"

def newpd_weather(file):    #this gives a new pd with Avg temp, Max temp, lownest temp(EMNT), rainfall (PRCP), Snow
    file = pd.read_csv(file) 
    file = file.loc[:,['DATE',"TAVG","TMAX", "EMNT", "PRCP","SNOW"]]
    file2 = file[file["DATE"] >= '1959-07']
    file3 = file2[file2['DATE'] <= '2023-07']
    file3.set_index('DATE', inplace=True)
    file3.index = pd.to_datetime(file3.index)
    
    return file3

def weather_predict(file, date):
  date = date[0:7]
  file = pd.read_csv(file)
  file = file.loc[:,['DATE',"TAVG","TMAX", "EMNT", "PRCP","SNOW"]]
  file2 = file[file["DATE"] >= date]
  file3 = file2[file2['DATE'] <= date]
  file3.set_index('DATE', inplace=True)
  file3.index = pd.to_datetime(file3.index)
  
  return file3
  
#print(newpd_weather(weather_data_dm))

def weather_combined(file1, file2, file3,file4):
  
  file1 = pd.read_csv(file1)
  file2 = pd.read_csv(file2)
  file3 = pd.read_csv(file3)
  file4 = pd.read_csv(file4)
  file1 = file1[file1['DATE']>="1959-07"]
  file1 = file1[file1['DATE']<="2017-02"]
  file2 = file2[file2['DATE']>="1959-07"]
  file2 = file2[file2['DATE']<="2017-02"]
  file3 = file3[file3['DATE']>="1959-07"]
  file3 = file3[file3['DATE']<="2017-02"]
  file4 = file4[file4['DATE']>="1959-07"]
  file4 = file4[file4['DATE']<="2017-02"]

  file1 = file1.loc[:,['DATE',"TAVG","TMAX", "EMNT", "PRCP","SNOW"]]
  file2 = file2.loc[:,['DATE',"TAVG","TMAX", "EMNT", "PRCP","SNOW"]]
  file3 = file3.loc[:,['DATE',"TAVG","TMAX", "EMNT", "PRCP","SNOW"]]
  file4 = file4.loc[:,['DATE',"TAVG","TMAX", "EMNT", "PRCP","SNOW"]]
  
  file1.set_index('DATE', inplace=True)
  file1.index = pd.to_datetime(file1.index)
  file2.set_index('DATE', inplace=True)
  file2.index = pd.to_datetime(file2.index)
  file3.set_index('DATE', inplace=True)
  file3.index = pd.to_datetime(file3.index)
  file4.set_index('DATE', inplace=True)
  file4.index = pd.to_datetime(file4.index)
  file1comb = file1.join(file2, how = "inner", rsuffix='2')
  file1comb = file1comb.join(file3, how= "inner", rsuffix='3')
  file1comb = file1comb.join(file4, how="inner", rsuffix="4")
  
  
  
  
  
  
  # file1comb = file1comb.drop('DATE2', axis = 1)
  # file1comb = file1comb.drop('DATE3', axis = 1)
  # file1comb = file1comb.drop('DATE4', axis = 1)
  file1comb = file1comb.interpolate('linear')
  
 
  
  
  return file1comb

def weather_predict_comb(date):
  file1 = "DesMoine_weather_data.csv"
  file2 = "Witchita.csv"
  file3 = "Tulsa_airport.csv"
  file4 = "Grandcentral_NE.csv"
  file1 = pd.read_csv(file1)
  file2 = pd.read_csv(file2)
  file3 = pd.read_csv(file3)
  file4 = pd.read_csv(file4)
  file1 = file1[file1['DATE']>="1959-07"]
  file1 = file1[file1['DATE']<="2024-02"]
  file2 = file2[file2['DATE']>="1959-07"]
  file2 = file2[file2['DATE']<="2024-02"]
  file3 = file3[file3['DATE']>="1959-07"]
  file3 = file3[file3['DATE']<="2024-02"]
  file4 = file4[file4['DATE']>="1959-07"]
  file4 = file4[file4['DATE']<="2024-02"]

  file1 = file1.loc[:,['DATE',"TAVG","TMAX", "EMNT", "PRCP","SNOW"]]
  file2 = file2.loc[:,['DATE',"TAVG","TMAX", "EMNT", "PRCP","SNOW"]]
  file3 = file3.loc[:,['DATE',"TAVG","TMAX", "EMNT", "PRCP","SNOW"]]
  file4 = file4.loc[:,['DATE',"TAVG","TMAX", "EMNT", "PRCP","SNOW"]]
  
  file1.set_index('DATE', inplace=True)
  file1.index = pd.to_datetime(file1.index)
  file2.set_index('DATE', inplace=True)
  file2.index = pd.to_datetime(file2.index)
  file3.set_index('DATE', inplace=True)
  file3.index = pd.to_datetime(file3.index)
  file4.set_index('DATE', inplace=True)
  file4.index = pd.to_datetime(file4.index)
  file1comb = file1.join(file2, how = "inner", rsuffix='2')
  file1comb = file1comb.join(file3, how= "inner", rsuffix='3')
  file1comb = file1comb.join(file4, how="inner", rsuffix="4")
  file1comb = file1comb.loc[[date]]
  
  
  
  
  
  
  # file1comb = file1comb.drop('DATE2', axis = 1)
  # file1comb = file1comb.drop('DATE3', axis = 1)
  # file1comb = file1comb.drop('DATE4', axis = 1)
  file1comb = file1comb.interpolate('linear')
 
  
 
  
  
  return file1comb
  

#print(weather_combined(file1, file2, file3, file4))
#print(weather_predict_comb('2024-01-01'))