######  Loading data

import datetime as dt
import datetime
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sqlalchemy import create_engine # database connection

# Creating the data base engine to create the database 
# Create this only once as it will throw an exception that it already exists later 
a=str('file_{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())+'.db')
disk_engine = create_engine('sqlite:///'+a)
#start = dt.datetime.now()

# creating new file to put log data of the code running
log_file=open("log_stats.txt","w")
log_file.write("############### Start #################\n")

# we increase the chunk size as the datasize 6 million
# but if u are running on the subset file "output_ir.csv" 
#then you can uncomment the other chunksize
#chunksize= 2000
chunksize = 180000
j = 0
index_start = 1

log_file.write("\nAdding to Database ....")
##===>>> run following lines of codes only once as it appends data to the database table "data" :
# here change the name of the file and the directory as per requirement
for df in pd.read_csv('Output_ir.csv', names=['Id', 'Title', 'Body', 'Tags'], \
                       chunksize=chunksize, iterator=True, encoding='utf-8', ):

	df.index += index_start

	j+=1
	print '{} rows'.format(j*chunksize)

	df.to_sql('data', disk_engine, if_exists='append')
	index_start = df.index[-1] + 1

log_file.write("\nFinished ...")

#establish connection to the above created database to read all rows
con = sqlite3.connect(a)
data = pd.read_sql_query("""SELECT Tags FROM data""", con)
# Close the connection established
con.close()

log_file.write("\n\n# of row in the dataset used : ")
log_file.write(str(data.shape[0]))
log_file.write("\n\n# of col in the dataset used : ")
log_file.write(str(data.shape[1]))

data.drop(data.index[0], inplace=True)
print data.info()
print data.describe()


######### count tags
#by default 'split()' will tokenize each tag using space.
log_file.write("\n\nTokenizing the tags to get their counts and stats ....")

# Countvectorizer is a model on sklearn which 
#helps get a histographic data for the input (frequency)

vectorizer = CountVectorizer(tokenizer = lambda x: x.split())
tag_dtm = vectorizer.fit_transform(data['Tags'])

tags = vectorizer.get_feature_names()
print "Number of unique tags = %d"%len(tags)
log_file.write("\n\nNumber of unique tags = %d"%len(tags))
#print tags[:10]

terms = vectorizer.get_feature_names()
freqs = tag_dtm.sum(axis=0).A1
result = dict(zip(terms, freqs))
result_pd=pd.DataFrame(result.items())

log_file.write("\n\n Writing to unique_tags_count.csv file ...")
with open('unique_tags_count.csv', 'wb') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in result.items():
        writer.writerow([key, value])

# the numbers below like 10000 and 100000 works for larger datasets
# but u can uncomment some lines and work it for larger dataset
tag_df = pd.read_csv("unique_tags_count.csv", names=['Tags', 'Counts'])
tag_df.head()
tag_df[tag_df.Counts>10000].head()
lst_tags_gt_10k = tag_df[tag_df.Counts>10000].Tags

'''
log_file.write('\n\n{} Tags are used more than 10000 times'.format(len(lst_tags_gt_10k)))
print '{} Tags are used more than 10000 times'.format(len(lst_tags_gt_10k))

lst_tags_gt_100k = tag_df[tag_df.Counts>100000].Tags
log_file.write('\n\n{} Tags are used more than 100000 times'.format(len(lst_tags_gt_100k)))
print '{} Tags are used more than 100000 times'.format(len(lst_tags_gt_100k))
'''

lst_tags_gt_100 = tag_df[tag_df.Counts>100].Tags
log_file.write('\n\n{} Tags are used more than 100 times'.format(len(lst_tags_gt_100)))
print '{} Tags are used more than 100 times'.format(len(lst_tags_gt_100))


#########  tags per question 
tag_count = tag_dtm.sum(axis=0).tolist()[0]
tag_count = [int(j) for j in tag_count]
tag_quest_count = tag_dtm.sum(axis=1).tolist()
tag_quest_count=[int(j) for i in tag_quest_count for j in i]

print "Maximum number of tags per question: %d"%max(tag_quest_count)
print "Minimum number of tags per question: %d"%min(tag_quest_count)
print "Avg. number of tags per question: %f"% ((sum(tag_quest_count)*1.0)/len(tag_quest_count))

log_file.write("\n\nMaximum number of tags per question: %d"%max(tag_quest_count))
log_file.write("\n\nMinimum number of tags per question: %d"%min(tag_quest_count))
log_file.write("\n\nAvg. number of tags per question: %f"% ((sum(tag_quest_count)*1.0)/len(tag_quest_count)))
#### plots histogram

log_file.write("\n\nPloting Histogram for tags.....")
x=[]
y=[]
for i in range(1,max(tag_quest_count)+1):
    x.append(i)
    y.append(tag_quest_count.count(i))


plt.title("Number of tags in the questions ")
plt.xlabel("Number of Tags")
plt.ylabel("Number of questions")
plt.plot(x,y,color='blue',marker='o',linestyle='--')
plt.show()

log_file.write("\n\nPloting Done .....")

####### top 10 tags with graph
log_file.write("\n\nPloting Top 10 tags.....")

tag_df.sort_values(['Counts'], ascending=False, inplace=True) 
i=np.arange(10)
tag_df.head(10).plot(kind='barh',color='green')
plt.title('Frequency of top 10 tags')
plt.yticks(i, tag_df['Tags'])
plt.xlabel('counts')
plt.ylabel('Tags')
plt.show()
log_file.write("\n\nPloting Done .....")
