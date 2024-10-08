# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 14:12:13 2023

@author: COMPUTER
"""

import pandas as pd
import numpy as np
#reading dataset
df=pd.read_csv('E:/project 2023/spam review detection/spam.csv',encoding='latin-1')
#checking size of dataframe
df.shape
df.head(5)
df.info()
#dropping un necessary columns
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
#renaming the columns
df.rename(columns={'lable':'target','text':'text'},inplace=True)
df.head(2)
df.to_csv("spam1.csv")

# with open('output.csv', 'w', encoding='utf-8') as file:
#     file.write(df)
# from sklearn.preprocessing import LabelEncoder
# encoder=LabelEncoder()
# df['target']=encoder.fit_transform(df['target'])

# df.head(2)

# #Checking missing values
# df.isnull().sum()
# #cheking duplicated values
# df.duplicated().sum()
# #dropping duplicates value
# print("before removing duplicates;",df.shape)
# df.drop_duplicates(keep='first',inplace=True)
# print("after removing duplicates",df.shape)

# #Checking counts of Ham and spam
# df['target'].value_counts().plot(kind='bar')

# import matplotlib.pyplot as plt
# plt.pie(df["target"].value_counts(),labels=['ham','spam'],autopct='%1.1f%%')
# plt.show()

# import nltk

# #num of characters
# df['num_characters'] = df['text'].apply(len)
# # num of words
# df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
# #num of sentences
# df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

# df.head(2)

# #overall(ham and spam)
# df[['num_characters','num_words','num_sentences']].describe()

# # ham
# df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()

# # spam
# df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()

# #num_characters
# import seaborn as sns
# sns.histplot(df[df['target'] == 0]['num_characters'], label='Non-spam')
# sns.histplot(df[df['target'] == 1]['num_characters'], label='Spam')

# plt.xlabel('Number of Characters')
# plt.ylabel('Frequency')
# plt.title('Histogram of Number of Characters')

# plt.legend()

# plt.show()

# #num_words
# sns.histplot(df[df['target'] == 0]['num_words'], label='Non-spam')
# sns.histplot(df[df['target'] == 1]['num_words'], label='Spam')

# plt.xlabel('Number of words')
# plt.ylabel('Frequency')
# plt.title('Histogram of Number of words')

# plt.legend()

# plt.show()

# # for num_sentences

# fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# # Plot for non-spam messages
# sns.histplot(df[df['target'] == 0]['num_sentences'], ax=axes[0])
# axes[0].set_xlabel('Number of Sentences')
# axes[0].set_ylabel('Frequency')
# axes[0].set_title('Histogram of Number of Sentences (ham)')

# # Plot for spam messages
# sns.histplot(df[df['target'] == 1]['num_sentences'], ax=axes[1])
# axes[1].set_xlabel('Number of Sentences')
# axes[1].set_ylabel('Frequency')
# axes[1].set_title('Histogram of Number of Sentences (Spam)')

# plt.tight_layout()
# plt.show()

# #finding correlation
# df.corr()

# #visualize correlation
# sns.heatmap(df.corr(),annot=True)