import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the downloaded mail data to the pandas dataframe
data = pd.read_csv('spam_data.csv', encoding='latin-1')

# Replacing the null values present in data with empty string
mailData = data.where((pd.notnull(data)), '')

# Setting spam mail to 1 and Not Spam mail to 0
mailData.loc[mailData['v1'] == 'spam', 'v1'] = 1
mailData.loc[mailData['v1'] == 'ham', 'v1'] = 0

# Separating the data into messages and category
message = mailData['v2']
category = mailData['v1']

# Splitting the data into training data and testing data
messageTrain, messageTest, categoryTrain, categoryTest = train_test_split(message, category, test_size=0.2, random_state=3)
featureCollection = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
messageTrainFeatures = featureCollection.fit_transform(messageTrain)
messageTestFeatures = featureCollection.transform(messageTest)

# Convert categoryTrain and categoryTest values to integers
categoryTrain = categoryTrain.astype('int')
categoryTest = categoryTest.astype('int')

# Applying Logistic regression
component = LogisticRegression()
component.fit(messageTrainFeatures, categoryTrain)

# Make predictions on the test data
predictions = component.predict(messageTestFeatures)

# Calculate accuracy
accuracy = accuracy_score(categoryTest, predictions)

# print("Accuracy:", accuracy)
testInputData = ["Mass and Energy are interchangeable under certain circumstances. When atoms split, the process is called nuclear fission."]

# Test 1
testDataFeatures1 = featureCollection.transform(testInputData)
judgement1 = component.predict(testDataFeatures1)
if judgement1[0] == 1:
    print("Sample 1: This is a Spam mail")
else:
    print("Sample 1: This is not a Spam mail")

# Test 2
testData2 = input("Enter a message to check if it's spam or not: ") 
testDataFeatures2 = featureCollection.transform([testData2])  
judgement2 = component.predict(testDataFeatures2)
if judgement2[0] == 1:
    print("Sample 2: This is a Spam mail")
else:
    print("Sample 2: This is not a Spam mail")
