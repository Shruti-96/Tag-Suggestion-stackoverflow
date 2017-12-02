######  Loading data

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import re
import nltk
import pandas as pd
import datetime
import warnings
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine # database connection

stemmer = PorterStemmer()
lmtzr = WordNetLemmatizer()

def get_tokens(text):

	lowers = text.lower()
	print(string.punctuation)
	tokens =text.split(" ")
	stems = stem_tokens(tokens,lmtzr)
	print(stems)
	return stems

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.lemmatize(item))
        #stemmed.append(stemmer.stem(item))
    return stemmed

# For filtering the common_words
def filter_common_words(text, stop_words):

    symbols = [',','.',':',';','+','=','"','/','?','(',')','!','$']
    for symbol in symbols:
        text = text.replace(symbol,' ')
    output = ""
    for word in text.split():
        if not word.lower() in stop_words:
            output += word + ' '
    return output

# filtering the html tags
def filter_html_tags(text):
    # the following tags and their content will be removed, for example <a> tag will remove any html links
    tags_to_filter = ['code','a']
    # if isinstance(text, unicode):
    #     text = text.encode('utf8')
    soup = BeautifulSoup(text,"lxml")
    for tag_to_filter in tags_to_filter:
        text_to_remove = soup.findAll(tag_to_filter)
        [tag.extract() for tag in text_to_remove]
    return soup.get_text()

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Creating the data base engine to create the database 
# Create this only once as it will throw an exception that it already exists later 
a=str('file_{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())+'.db')
disk_engine = create_engine('sqlite:///'+a)
#start = dt.datetime.now()

# creating new file to put log data of the code running
log_file=open("log_main.txt","w")
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

###### Preprocessing
log_file.write("\n\nStarting Preprocessing ....")

stop_words = {'why', 'as', 'other', 'couldn', 'were', 'such', 'itself', 'yours', 'until', 'shan', 'been', 'just', 'i', 'themselves', 'not', 't', 'ours', 'doesn', 'it', 'own', 'below', 'during', 'd', 'my', 'up', 'hadn', 'our', 'between', 'he', 'shouldn', 'wasn', 'at', 'do', 'yourself', 'm', 'into', 'to', 'wouldn', 'down', 'how', 'or', 'that', 'above', 'through', 'out', 'o', 'ain', 'your', 'here', 'ourselves', 'more', 'aren', 'from', 'himself', 'doing', 'for', 'too', 'the', 'now', 'only', 'both', 'some', 'while', 'me', 're', 'very', 'and', 'because', 'ma', 'what', 'you', 'being', 'have', 'before', 'then', 'where', 'under', 'are', 'same', 'further', 'his', 'is', 'didn', 'll', 'but', 've', 'again', 'can', 'whom', 'with', 'there', 'hers', 's', 'of', 'won', 'once', 'these', 'no', 'an', 'haven', 'needn', 'which', 'its', 'should', 'them', 'those', 'when', 'we', 'has', 'yourselves', 'hasn', 'she', 'am', 'than', 'by', 'did', 'if', 'weren', 'who', 'does', 'herself', 'in', 'off', 'nor', 'any', 'against', 'her', 'theirs', 'a', 'after', 'they', 'all', 'myself', 'so', 'on', 'each', 'had', 'about', 'their', 'few', 'him', 'be', 'isn', 'mightn', 'this', 'most', 'y', 'over', 'was', 'having', 'will', 'mustn', 'don'}


stemmer = SnowballStemmer("english")
#function to remove html tags from our document.
def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

con = sqlite3.connect(a)

reader = con.cursor()
writer = con.cursor()
reader.execute("SELECT Title, Body, Tags From data LIMIT 1000")
log_file.write("\n\nCreating the table QuestionProcessed3 ....")
writer.execute("create table IF NOT EXISTS QuestionsProcessed3(question text, code text, Tags text, words_pre int, words_post int, has_code int)")
reader.fetchone()
no_dup_code=0
len_pre=0L
len_post=0L
i=0
log_file.write("\n\nRemoving Code Portion, HTML Tags & stopwords/punctuations ....")
for row in reader:
    is_code = 0
    title=row[0]
    question=row[1]
   # print question
    tags=row[2]
    var=tags
    x=len(question)+len(title)
    if '<code>' in question:
        no_dup_code+=1
        is_code = 1
    len_pre+=x
    code = str(re.findall(r'<code>(.*?)</code>', question, flags=re.DOTALL))
    question=re.sub('<code>(.*?)</code>', '', question, flags=re.MULTILINE|re.DOTALL)
    question=question.encode('utf-8')
    title=title.encode('utf-8')
    var=var.encode('utf-8')
    question=striphtml(question)
    #Removing all non-alphabet characters from question
    if i<=80000:
        question=title+" "+title+" "+title+" "+question+" "+var
    else:
        question=title+" "+title+" "+title+" "+question
    i+=0
    question=re.sub(r'[^A-Za-z0-9#+.\-]+',' ',question)
    question=str(question.lower())
    
    #Removing all single letter and and stopwords from question exceptt for the letter 'c'
    question=' '.join(str(stemmer.stem(j)) for j in question.split() if j not in stop_words)
    
    len_post+=len(question)
    tup = (question,code,tags,x,len(question))
    writer.execute("insert into QuestionsProcessed3(question,code,Tags,words_pre,words_post) values (?,?,?,?,?)",tup)
con.commit()
con.close()

log_file.write("\nFinished ...")

#code to show that the top500 tags cover 90% of the data
con = sqlite3.connect(a)
df = pd.read_sql_query("""SELECT question, Tags FROM QuestionsProcessed3""", con)
con.close()
df.shape


vect = CountVectorizer(tokenizer = lambda x: x.split())
tag_dtm = vect.fit_transform(df['Tags'])

tag_dtm
tags=vect.get_feature_names()

t = tag_dtm.sum(axis=0).tolist()[0]
sorted_tags_i = sorted(range(len(t)), key=lambda i: t[i], reverse=True)
tag_500=tag_dtm[:,sorted_tags_i[:500]]


########## train and test split data

total_size=len(df)
train_size=int(0.80*total_size)
x_train=df.head(train_size)
x_test=df.tail(total_size - train_size)
y_train = tag_500[0:train_size,:]
y_test = tag_500[train_size:total_size,:]

########## Feature vectorizing part 
log_file.write("\n\nRunning the TF-IDF for feature extration....")

vectorizer = TfidfVectorizer(min_df=0.00009, max_features=200000, smooth_idf=True, norm="l2", \
                             tokenizer = lambda x: x.split(), sublinear_tf=False, ngram_range=(1,3))
question_dtm = vectorizer.fit_transform(x_train['question'])

question_dtm

test_question_dtm = vectorizer.transform(x_test['question'])

log_file.write("\n\nFinished ....")


############## classifier
log_file.write("\n\nStarting the classifier part....")
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression

log_file.write("\n\nSGD Classifier ....")
classifier = OneVsRestClassifier(SGDClassifier(loss='log', alpha=0.00001, verbose =0, penalty='l1'), n_jobs=-1)
classifier.fit(question_dtm, y_train)

predictions = classifier.predict(test_question_dtm)

precision = precision_score(y_test, predictions, average='micro')
recall = recall_score(y_test, predictions, average='micro')
f1 = f1_score(y_test, predictions, average='micro')
print("SGD classifier") 
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
log_file.write("\n\nPrecision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))


classifier_2 = OneVsRestClassifier(svm.LinearSVC(verbose =0), n_jobs=-1)
classifier_2.fit(question_dtm, y_train)
predictions_2 = classifier_2.predict(test_question_dtm)

precision = precision_score(y_test, predictions_2, average='micro')
recall = recall_score(y_test, predictions_2, average='micro')
f1 = f1_score(y_test, predictions_2, average='micro')
print("Linear SVC classifier")  
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
log_file.write("\n\nPrecision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

classifier_3 = OneVsRestClassifier(LogisticRegression(penalty='l1',verbose =0), n_jobs=-1)
classifier_3.fit(question_dtm, y_train)
predictions_3 = classifier_3.predict(test_question_dtm)

precision = precision_score(y_test, predictions_3, average='micro')
recall = recall_score(y_test, predictions_3, average='micro')
f1 = f1_score(y_test, predictions_3, average='micro')
print("Logistic Regression classifier") 
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
log_file.write("\n\nPrecision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

log_file.write("\n\nFinished ....")

