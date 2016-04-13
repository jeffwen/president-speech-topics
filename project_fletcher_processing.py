from bs4 import BeautifulSoup
from datetime import datetime, date, time
from nltk.corpus import stopwords
from collections import defaultdict
from babel.dates import format_date, format_datetime, format_time
from dateutil import parser
import nltk.stem
import pandas as pd
import requests
import numpy as np
import pymongo
from progressbar import ProgressBar
import re
import string
import pickle
import math

##########################
## Connect with MongoDB ##    
##########################

try:
    conn=pymongo.MongoClient()
    print "Connected successfully!"
except pymongo.errors.ConnectionFailure, e:
   print "Could not connect to MongoDB: %s" % e

db = conn['project_fletcher']
collection = db.speeches

# return all data
data_cursor = collection.find()
data = [i for i in data_cursor]

# separate the text only from each of the observations
data_text = [i['text'] for i in data]


# remove stop words and short sentences from the data
def preprocess_documents(collection_of_documents):
    '''
    input: text from president speeches
    output: list of lists of the text from paragraph speeches
    '''
    new_document_list = []
    stop_words = stopwords.words('english')
    exclude = set(string.punctuation)
    pbar = ProgressBar()
    for document in pbar(data_text):
        new_paragraph = []
        for paragraph in document:
            paragraph_words = paragraph.split()
            para = []

            # only keep if the paragraph is more than 10 words
            if len(paragraph_words) >= 10:
                for word in paragraph_words:
                    word = ''.join(ch for ch in word if ch not in exclude).lower()
                    # word =  nltk.stem.SnowballStemmer('english').stem(word)

                    # keep word if it is greater than 3 letters 
                    if (word not in stop_words) and len(word) > 3:
                        para.append(word)

            # checking the paragraph length again
            if len(para) >= 10:
                new_paragraph.append(para)
        new_document_list.append(new_paragraph)
    return new_document_list

# new_data_text = preprocess_documents(data_text)

# with open('clean_documents.pkl', 'w') as picklefile:
#     pickle.dump(new_data_text, picklefile)

# with open('clean_documents.pkl', 'rb') as picklefile:
#     data = pickle.load(picklefile)

# return a list of president names
pres = [i['president'] for i in data]


# create a dictionary with president as keys and speeches as paragraphs
def gen_pres_dict(presidents):
    '''
    input: president names
    output: dictionay with key as president name and value as the text from speeches
    '''
    presDict = {}
    pbar = ProgressBar()
    for pres in pbar(presidents):
        data_cursor = collection.find({"president":pres})
        presDict[pres] = [i for i in data_cursor]
    return presDict

presDict = gen_pres_dict(pres)

# with open('presDict.pkl', 'w') as picklefile:
#     pickle.dump(presDict, picklefile)

# with open('presDict.pkl', 'rb') as picklefile:
#     presDict = pickle.load(picklefile)


# create a cleaned version of the previous dictionary with value as a list of strings of text from the speeches
def clean_pres_dict(president_speech_dictionary):
    clean_pres_dict = {}

    # remove stopwords and punctuation
    stop_words = stopwords.words('english') + ['united', 'states','public','government','think','president','also','would','going']
    exclude = set(string.punctuation)
    
    presidents = president_speech_dictionary.keys()
    pbar = ProgressBar()
    for pres in pbar(presidents):
        pres_speech_list = []
        for speech in president_speech_dictionary[pres]:
            new_speech = []
            for paragraph in speech['text']:
                paragraph_words = paragraph.split()

                # remove short paragraphs
                if len(paragraph_words) >= 10:
                    for word in paragraph_words:
                        word = ''.join(ch for ch in word if ch not in exclude).lower()

                        # remove short words
                        if (word not in stop_words) and len(word) > 3:
                            new_speech += [word]
            new_speech = ' '.join(new_speech)
            pres_speech_list.append(new_speech)
        clean_pres_dict[pres] = pres_speech_list
    return clean_pres_dict

# dictionary of presdients as keys and speech texts as list of strings
presidentDict = clean_pres_dict(presDict)

# create condensed one string for each president
def condense_speeches(aDict):
    pres_names = sorted(aDict.keys())
    new_dict = []
    for name in pres_names:
        new_dict.append(reduce(lambda x,y: x+y, aDict[name]))
    return new_dict

# list of strings with each string as total of all the presdients' speeches
new_dict = condense_speeches(presidentDict)

# def make_pred_string(aDict):
#     newDict = defaultdict(list)
#     for pres in aDict.keys():
#         for speech in aDict[pres]:
#             newDict[pres].append([' '.join(speech)])
#     return newDict

# newPresidentDict = make_pred_string(presidentDict)


# create a dicitonary with key as the president name and the value as a list of individual words that the president used in the speeches
def clean_pres_dict_2(president_speech_dictionary):
    clean_pres_dict = {}
    stop_words = stopwords.words('english') + ['united', 'states','public','government','think','president','also','would','going']
    exclude = set(string.punctuation)
    presidents = president_speech_dictionary.keys()
    pbar = ProgressBar()
    for pres in pbar(presidents):
        pres_speech_list = []
        for speech in president_speech_dictionary[pres]:
            new_speech = []
            for paragraph in speech['text']:
                paragraph_words = paragraph.split()
                if len(paragraph_words) >= 10:
                    for word in paragraph_words:
                        word = ''.join(ch for ch in word if ch not in exclude).lower()
                        if (word not in stop_words) and len(word) > 3:
                            new_speech += [word]
            pres_speech_list.append(new_speech)
        clean_pres_dict[pres] = pres_speech_list
    return clean_pres_dict

# list of individual tokens for each president (speeches separated into words)
presidentDict_2 = clean_pres_dict_2(presDict)


# group by era
group_1 = ['George Washington', 'John Adams', 'Thomas Jefferson',
           'James Madison', 'James Monroe', 'John Quincy Adams']

group_2 = ['Andrew Jackson', 'Martin Van Buren', 'William Henry Harrison',
           'John Tyler', 'James K. Polk', 'Zachary Taylor', 'Millard Fillmore']

group_3 = ['Franklin Pierce', 'James Buchanan', 'Abraham Lincoln',
           'Andrew Johnson', 'Ulysses S. Grant', 'Rutherford B. Hayes']

group_4 = ['James A. Garfield', 'Chester A. Arthur', 'Grover Cleveland',
           'Benjamin Harrison', 'Grover Cleveland']

group_5 = ['William McKinley','Theodore Roosevelt', 'William Howard Taft',
           'Woodrow Wilson']

group_6 = ['Warren G. Harding', 'Calvin Coolidge', 'Herbert Hoover',
           'Franklin D. Roosevelt', 'Harry S. Truman', 'Dwight D. Eisenhower']

group_7 = ['John F. Kennedy', 'Lyndon B. Johnson', 'Richard Nixon',
           'Gerald R. Ford', 'Jimmy Carter', 'Ronald Reagan']

group_8 = ['George Bush','William J. Clinton', 'George W. Bush', 'Barack Obama']


# generate by paragraph list by era and keep track of when each speech took place so that I could call the info later
def pres_by_era_dict(president_speech_dictionary,group):
    clean_pres_dict = {}
    stop_words = stopwords.words('english')# + ['united', 'states','public','government','think','president','also','would','going']
    exclude = set(string.punctuation)
    presidents = [i for i in president_speech_dictionary.keys() if i in group]
    pbar = ProgressBar()
    for pres in pbar(presidents):
        pres_speech_list = []
        new_speech = []
        speech_count = -1
        count_list = []
        original_paragraph_list = []
        title_date = []
        for speech in president_speech_dictionary[pres]:
            speech_count += 1
            para_count = -1
            title = speech['title']
            date = speech['date']
            for paragraph in speech['text']:
                para_count += 1
                new_paragraph = []
                paragraph_words = paragraph.split()
                if len(paragraph_words) >= 10:
                    for word in paragraph_words:
                        word = ''.join(ch for ch in word if ch not in exclude).lower()
                        if (word not in stop_words) and len(word) > 3:
                            new_paragraph += [word]
                if new_paragraph != []:
                    new_speech.append(' '.join(new_paragraph))
                    count_list += [(pres, str(speech_count)+'_'+str(para_count))]
                    original_paragraph_list += [paragraph]
                    title_date += [(date,title)]
        clean_pres_dict[pres] = {'speech':new_speech,'paragraph_location':count_list, 'original_paragraph':original_paragraph_list, 'title_date':title_date}
    return clean_pres_dict

# dictionary of presdients as keys and speech texts as list of strings by group
group_1_dict = pres_by_era_dict(presDict, group_1)
group_2_dict = pres_by_era_dict(presDict, group_2)
group_3_dict = pres_by_era_dict(presDict, group_3)
group_4_dict = pres_by_era_dict(presDict, group_4)
group_5_dict = pres_by_era_dict(presDict, group_5)
group_6_dict = pres_by_era_dict(presDict, group_6)
group_7_dict = pres_by_era_dict(presDict, group_7)
group_8_dict = pres_by_era_dict(presDict, group_8)

# join sentences for each president into long list of strings to be vectorized
def period_condenser(someDict):
    keys = sorted(someDict.keys())
    total_list = []
    paragraph_location = []
    original_paragraph = []
    title_date = []
    president = []
    pbar = ProgressBar()
    for pres in pbar(keys):
        total_list += someDict[pres]['speech']
        paragraph_location += someDict[pres]['paragraph_location']
        original_paragraph += someDict[pres]['original_paragraph']
        title_date += someDict[pres]['title_date']
        president += [pres]*len(someDict[pres]['speech'])
    return total_list, paragraph_location, original_paragraph, title_date, president

# sentences are sorted based on alphabetical order of president names
# things are quite messy here because I wanted to keep track of not only the speech but the index of the particular paragraph so that I could bring it up on the front end that I was building
group_1_speeches, group_1_location, group_1_original_paragraph, g_1_title_date, g_1_pres = period_condenser(group_1_dict)
group_2_speeches, group_2_location, group_2_original_paragraph, g_2_title_date, g_2_pres = period_condenser(group_2_dict)
group_3_speeches, group_3_location, group_3_original_paragraph, g_3_title_date, g_3_pres = period_condenser(group_3_dict)
group_4_speeches, group_4_location, group_4_original_paragraph, g_4_title_date, g_4_pres = period_condenser(group_4_dict)
group_5_speeches, group_5_location, group_5_original_paragraph, g_5_title_date, g_5_pres = period_condenser(group_5_dict)
group_6_speeches, group_6_location, group_6_original_paragraph, g_6_title_date, g_6_pres = period_condenser(group_6_dict)
group_7_speeches, group_7_location, group_7_original_paragraph, g_7_title_date, g_7_pres = period_condenser(group_7_dict)
group_8_speeches, group_8_location, group_8_original_paragraph, g_8_title_date, g_8_pres = period_condenser(group_8_dict)

# with open('speeches_and_locations.pkl', 'w') as picklefile:
#     pickle.dump((group_1_speeches, group_1_location, group_1_original_paragraph, g_1_title_date), picklefile)
#     pickle.dump((group_2_speeches, group_2_location, group_2_original_paragraph, g_2_title_date), picklefile)
#     pickle.dump((group_3_speeches, group_3_location, group_3_original_paragraph, g_3_title_date), picklefile)
#     pickle.dump((group_4_speeches, group_4_location, group_4_original_paragraph, g_4_title_date), picklefile)
#     pickle.dump((group_5_speeches, group_5_location, group_5_original_paragraph, g_5_title_date), picklefile)
#     pickle.dump((group_6_speeches, group_6_location, group_6_original_paragraph, g_6_title_date), picklefile)
#     pickle.dump((group_7_speeches, group_7_location, group_7_original_paragraph, g_7_title_date), picklefile)
#     pickle.dump((group_8_speeches, group_8_location, group_8_original_paragraph, g_8_title_date), picklefile)


# # clean pres dict 2 with only nouns
# def clean_pres_dict_3(president_speech_dictionary):
#     clean_pres_dict = {}
#     stop_words = stopwords.words('english') + ['united', 'states','public','government','think','president','also','would','going']
#     exclude = set(string.punctuation)
#     presidents = president_speech_dictionary.keys()
#     pbar = ProgressBar()
#     for pres in pbar(presidents):
#         pres_speech_list = []
#         for speech in president_speech_dictionary[pres]:
#             new_speech = []
#             for paragraph in speech['text']:
#                 paragraph_words = nltk.pos_tag(paragraph.split())
#                 paragraph_words = [i[0] for i in paragraph_words if i[1] == 'NN' or i[1] == 'NNP' or i[1] == 'NNS' or i[1] == 'NNPS']
#                 # print paragraph_words
#                 if len(paragraph_words) >= 10:
#                     for word in paragraph_words:
#                         word = ''.join(ch for ch in word if ch not in exclude).lower()
#                         if (word not in stop_words) and len(word) > 3:
#                             new_speech += [word]
#                 else:
#                     continue
#                 # print new_speech
#             pres_speech_list.append(new_speech)
#             # print pres_speech_list
#         clean_pres_dict[pres] = pres_speech_list
#     return clean_pres_dict

# # list of individual tokens for each president (speeches separated)
# presidentDict_3 = clean_pres_dict_3(presDict)



###################
## Gensim Models ##
###################

from gensim import corpora, models
from sklearn.feature_extraction.text import CountVectorizer
from gensim.matutils import Sparse2Corpus, corpus2dense
import logging
from scipy import spatial
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# creating count vectorizer object so that documents can be vectorized for speeches (for LDA model and LSI model)
count_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,2), stop_words='english',token_pattern='\\b[a-z][a-z]+\\b')

# testing the count vectorizer for a single president
count_vects = count_vectorizer.fit_transform(presidentDict['George Washington']).transpose()

# creating a count to word mapping for LDA and LSI in gensim
id2word = dict((v,k) for k,v in count_vectorizer.vocabulary_.iteritems())

# LDA
corpus = Sparse2Corpus(count_vects)
ldamodel = models.ldamodel.LdaModel(corpus, num_topics=10, id2word = id2word, passes=20)

ldamodel.print_topics(10)
a = [i for i in corpus]
doc_topic = ldamodel[a[0]]

# LSI
tfidf = models.TfidfModel(corpus)
tfidf_corpus = tfidf[corpus]
lsi = models.LsiModel(tfidf_corpus, id2word=id2word, num_topics=10)
lsi_vecs = lsi[tfidf_corpus]
docs = [doc for doc in lsi_vecs]
lsi_np_vecs = corpus2dense(lsi_vecs, num_terms=100).transpose()


# create similarity matrix (cosine similarity)
def sim_matrix(lsi_output):
    sim_scores = []
    for i in range(len(lsi_np_vecs)):
        rows = []
        for j in range(len(lsi_np_vecs)):
            rows += [1.0 - spatial.distance.cosine(lsi_np_vecs[i], lsi_np_vecs[j])]
        sim_scores.append(rows)
    return sim_scores

df = pd.DataFrame(sim_matrix, columns = sorted(presidentDict.keys()), index = sorted(presidentDict.keys()))

df.to_csv('similarity_scores.csv')

# generate the edges based on a threshold of >= 0.75 cosine similarity
def create_edges(dataframe):
    columns = dataframe.columns
    aDict = {}
    for i in columns:
        for j in columns:
            if i!=j and dataframe.ix[i].ix[j]>=0.75:
                aDict['_'.join(set([i,j]))] = dataframe.ix[i].ix[j]
    return aDict

new_df = create_edges(df)
new_df = pd.DataFrame([new_df]).transpose()

source_target = map(lambda x: x.split('_'),new_df.index)
new_df['source'] = [i[0] for i in source_target]
new_df['target'] = [i[1] for i in source_target]

new_df.to_csv('source_target.csv')


###################
## Period Topics ##
###################


from sklearn.feature_extraction import text 

# packaging all processes into one function (count vectorizing, building LDA model, extracting topics)
def gen_period_topics(list_by_period):
    '''
    input: presidents' speeches within the same period as a list
    output: top topics within those presidents' speeches
    '''
    # remove stop words
    stop_words = text.ENGLISH_STOP_WORDS.union(['united','states','object','great','congress','president','like','time','said','going','government','good','dont','think','thing','country','things','know','american','regard','program','question','public','shall','people','summit','helen','didnt','youre','talking','comment','tell','service','theyre','want','thats','make'])

    # count vectorize the speeches
    count_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,2), stop_words=stop_words,max_df=0.60, token_pattern='\\b[a-z][a-z]+\\b')
    count_vects = count_vectorizer.fit_transform(list_by_period).transpose()
    
    id2word = dict((v,k) for k,v in count_vectorizer.vocabulary_.iteritems())
    # LDA
    corpus = Sparse2Corpus(count_vects)
    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=10, id2word = id2word, passes=20)
    # print ldamodel.print_topics(10)
    a = [i for i in corpus]
    docs_topics = []
    pbar = ProgressBar()
    for doc in pbar(a):
        topic_ids = ldamodel.get_topic_terms(ldamodel[doc][0][0])
        topic_words = [id2word[i[0]] for i in topic_ids]
        docs_topics.append([ldamodel[doc][0][0],tuple(topic_words[:5])])
    return docs_topics

g_1_doc_topics = gen_period_topics(group_1_speeches)
g_2_doc_topics = gen_period_topics(group_2_speeches)
g_3_doc_topics = gen_period_topics(group_3_speeches)
g_4_doc_topics = gen_period_topics(group_4_speeches)
g_5_doc_topics = gen_period_topics(group_5_speeches)
g_6_doc_topics = gen_period_topics(group_6_speeches)
g_7_doc_topics = gen_period_topics(group_7_speeches)
g_8_doc_topics = gen_period_topics(group_7_speeches)


# count how many times a set of topics appears
def topic_counter(topic_list):
    counter = defaultdict(int)
    for aList in topic_list:
        counter[tuple(aList)] += 1
    return [(i[0],i[1],j) for i,j in counter.items()]

# scaler for the size of D3 circles 
def scaler(df_column):
    old_range = max(df_column)-min(df_column)*1.0
    new_range = 7000*1.0-300
    new_area = []
    min_old = min(df_column)
    for val in df_column:
        new_area.append(((val*1.0-min_old)/old_range)*new_range+300*1.)
    return new_area

# test_counter_df = pd.DataFrame(test_counter)

# create topic count df
def create_count_df(doc_topics, period):
    topic_counts = topic_counter(doc_topics)
    df_counts = pd.DataFrame(topic_counts)
    df_counts.columns = ['topic_group','topic_words','counts']
    df_counts['period'] = period
    df_counts.sort_values(['counts'], ascending = False, inplace = True)
    df_counts = df_counts.iloc[:10,]
    df_counts['topic_words'] = df_counts['topic_words'].map(lambda x:', '.join(x))
    df_counts['scaled_counts'] = scaler(df_counts['counts'])
    df_counts['radius'] = df_counts['scaled_counts']
    df_counts['radius'] = df_counts['radius'].map(lambda x: math.sqrt(x/math.pi))
    return df_counts

# g_1_topic_count_df = create_count_df(g_1_doc_topics,1)
# g_2_topic_count_df = create_count_df(g_2_doc_topics,2)
# g_3_topic_count_df = create_count_df(g_3_doc_topics,3)
# g_4_topic_count_df = create_count_df(g_4_doc_topics,4)
# g_5_topic_count_df = create_count_df(g_5_doc_topics,5)
# g_6_topic_count_df = create_count_df(g_6_doc_topics,6)
# g_7_topic_count_df = create_count_df(g_7_doc_topics,7)
# g_8_topic_count_df = create_count_df(g_8_doc_topics,8)

# all_topic_count_df = pd.concat([g_1_topic_count_df,g_2_topic_count_df,g_3_topic_count_df,g_4_topic_count_df,g_5_topic_count_df,g_6_topic_count_df,g_7_topic_count_df,g_8_topic_count_df], ignore_index = True)


# create original paragraph df, extracting the original paragraph (pre cleaning) to be displayed on the front end
def create_orig_paragraph_df(doc_topics, pres_list, title_date, orig_paragraphs, period):
    topic_numbers = [doc[0] for doc in doc_topics]
    dates, titles = zip(*title_date)
    dates = [format_date(parser.parse(str(i)), "MMMM dd, Y", locale='en') for i in dates]
    period = [period]*len(orig_paragraphs)
    zipped_list = zip(period, topic_numbers , dates, pres_list, titles, orig_paragraphs)
    df_orig_para = pd.DataFrame(zipped_list)
    df_orig_para.columns = ['period','topic_group','date','president','title','paragraph']
    return df_orig_para


# running all the processing at once....
g_1_doc_topics = gen_period_topics(group_1_speeches)
g_2_doc_topics = gen_period_topics(group_2_speeches)
g_3_doc_topics = gen_period_topics(group_3_speeches)
g_4_doc_topics = gen_period_topics(group_4_speeches)
g_5_doc_topics = gen_period_topics(group_5_speeches)
g_6_doc_topics = gen_period_topics(group_6_speeches)
g_7_doc_topics = gen_period_topics(group_7_speeches)
g_8_doc_topics = gen_period_topics(group_7_speeches)

g_1_topic_count_df = create_count_df(g_1_doc_topics,1)
g_2_topic_count_df = create_count_df(g_2_doc_topics,2)
g_3_topic_count_df = create_count_df(g_3_doc_topics,3)
g_4_topic_count_df = create_count_df(g_4_doc_topics,4)
g_5_topic_count_df = create_count_df(g_5_doc_topics,5)
g_6_topic_count_df = create_count_df(g_6_doc_topics,6)
g_7_topic_count_df = create_count_df(g_7_doc_topics,7)
g_8_topic_count_df = create_count_df(g_8_doc_topics,8)

g_1_paragraph_df = create_orig_paragraph_df(g_1_doc_topics, g_1_pres, g_1_title_date, group_1_original_paragraph,1)
g_2_paragraph_df = create_orig_paragraph_df(g_2_doc_topics, g_2_pres, g_2_title_date, group_2_original_paragraph,2)
g_3_paragraph_df = create_orig_paragraph_df(g_3_doc_topics, g_3_pres, g_3_title_date, group_3_original_paragraph,3)
g_4_paragraph_df = create_orig_paragraph_df(g_4_doc_topics, g_4_pres, g_4_title_date, group_4_original_paragraph,4)
g_5_paragraph_df = create_orig_paragraph_df(g_5_doc_topics, g_5_pres, g_5_title_date, group_5_original_paragraph,5)
g_6_paragraph_df = create_orig_paragraph_df(g_6_doc_topics, g_6_pres, g_6_title_date, group_6_original_paragraph,6)
g_7_paragraph_df = create_orig_paragraph_df(g_7_doc_topics, g_7_pres, g_7_title_date, group_7_original_paragraph,7)
g_8_paragraph_df = create_orig_paragraph_df(g_8_doc_topics, g_8_pres, g_8_title_date, group_8_original_paragraph,8)

all_topic_count_df = pd.concat([g_1_topic_count_df,g_2_topic_count_df,g_3_topic_count_df,g_4_topic_count_df,g_5_topic_count_df,g_6_topic_count_df,g_7_topic_count_df,g_8_topic_count_df], ignore_index = True)

all_paragraph_df = pd.concat([g_1_paragraph_df,g_2_paragraph_df,g_3_paragraph_df,g_4_paragraph_df,g_5_paragraph_df,g_6_paragraph_df,g_7_paragraph_df,g_8_paragraph_df], ignore_index = True)

all_topic_count_df.to_csv('all_topic_count.csv')
all_paragraph_df.to_csv('all_paragraph.csv')

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
