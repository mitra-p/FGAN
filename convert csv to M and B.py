# import panadas for use dataframe
import pandas as pd 


#  read csv
df = pd.read_csv(r'D:\A.csv')

#  select specific row from dataframe
select = df.loc[df['label'] == 0]
select1= df.loc[df['label'] == 1]

# export csv file from dataframe
select.to_csv('D:\A0.csv')
select1.to_csv('D:\A1.csv')
