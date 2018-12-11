import numpy
import pandas as pd
import json
import re

article_read = pd.read_csv('NDAQ.csv', delimiter=',', names = ['date', 'open', 'high', 'low', 'close', 'adj_close','volume'])
#print(article_read)

data = article_read.loc[1:,'date':'close']

print(data)
dates = data.loc[:,'date']
# print(dates)

# print(data.loc[data['date'] == '2007-01-03'])


with open('titles.txt') as json_data1:
    dict_news = json.load(json_data1)

excel_dates = []  #contains all the dates from excel sheet

for d in dates:
    excel_dates.append(re.sub('-','',d))
    # if re.sub('-','',d) not in dict_news:
    #     print(d)

print("--------------------------------------------")

# count = 0
news_dates = list(dict_news.keys()) #contains all the dates from the news dataset

# for k in dict_news.keys():
#     if k not in excel_dates:
#         print(k)
        # count += 1




commondates = list(set(excel_dates) & set(news_dates))
print(commondates)

#
#
# with open('titles.txt') as json_data1:
#     dict_news = json.load(json_data1)
#
# print(dict_news['20120908'])



