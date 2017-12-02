----------------
###   TITLE  ###
----------------
		Tag Suggestion for Stackoverflow Platform



------------------------------------------
###  Authors (Group Number : 8)       ####
------------------------------------------
		Badadhe Shruti Ekanath – CS14BTECH11005
		Shruti Bhatambre - CS14BTECH11007
		Krishna Priya – CS14BTECH11010



-----------------------------------
###       Files Submited        ###
-----------------------------------
		1) results.py
		2) stats.py
		3) readme.md
		4) ppt
		5) Output_ir.csv
		6) log_stats.txt
		7) log_main.txt
		8) unique_tags_count.csv





---------------------------------------
###        EXECUTION DETAILS        ###
---------------------------------------
		In results.py we have modified the code (We have reduced the data set to 10000 posts) to make it run in considerable amount of time in the local PCs

		The results we presented were on larger dataset. To run on larger dataset please uncomment respective parts in results.py file. To compute on larger dataset we ran on IITH GPU server. 




------------------------------------------------------------
###      REQUIREMENTS       ###
------------------------------------------------------------
		1.  sklearn
		2.  pandas
		3.  sqlite3
		4.  numpy
		5.  matplotlib
		6.  sqlalchemy
		7.  csv
		8.  nltk
		9.  re
		10. python2
		11. beautifulSoup
		12. datetime

To install any of the above dependencies execute:
	pip install <package name>



Full dataset is available on:
Data set : https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction


-----------------------------
###      How to run     ###
-----------------------------
1.) To get precision, recall values run the following command on terminal using python2
		python2 results.py

#outputs:
	1. prints output in log_main.txt and on terminal
	2. sgd classifier, linear svm classifier, logistic regression claasifier's f1 score, Precision, Recall



2.) To get statistics run the following commands on terminal using 
		python2 stats.py

#output:
	1. prints output in log_stats.txt and on terminal.
	2. outputs two graphs:
		1. tags vs number of questions histogram
		2. Top ten frequent tags



