import email
from multiprocessing import context
from django.shortcuts import render, redirect,get_object_or_404
from django.urls import reverse
from django.http import HttpResponse,HttpResponseRedirect, JsonResponse
from .models import *
from .forms import *
from datetime import datetime
from datetime import date, timedelta
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm
from django.db.models import Q
import os
from django.conf import settings
# Create your views here.

#csv bago toh
import csv

#INTEGRATE TOPIC MODELING
#Step 3
import numpy as np
import pandas as pd
from io import StringIO 

#NLTK
import nltk
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet 
#!pip install contractions
import contractions
#from textblob import Word

#Gensim
#!pip install gensim
import gensim
from gensim.parsing.preprocessing import STOPWORDS
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases
from gensim.test.utils import datapath
from gensim import models
from gensim.models import TfidfModel

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io, base64
from matplotlib.ticker import FuncFormatter
from collections import Counter
import matplotlib.colors as mcolors

#word cloud
#!pip install word_cloud
from wordcloud import WordCloud


import os

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)



#FUNCTIONS FOR DATA CLEANING
#Step 3 -Tokenization
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

#print(data_words[:1])

#Step 4 - Lemmarization
lemmatizer = WordNetLemmatizer()
def lemmatization(inputs):
    text_out = []
    for sent in inputs:
        lem = [lemmatizer.lemmatize(word=x, pos='v') for x in sent]
        text_out.append(lem)
    return text_out

def format_topics_sentences(ldamodel, corpus, texts):
    sent_topics_df = pd.DataFrame()
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: x[1], reverse=True) 
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contributions', 'Topic_Keywords']
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

def frequency_plot(topic):
    doc_lens = [len(d) for d in topic.Text]
    # Plot
    fig = plt.figure(figsize=(16,7), dpi=300)
    plt.hist(doc_lens, bins = 600, color='navy')
    plt.gca().set(xlim=(0, 600), ylabel='Number of Documents', xlabel="Document Word Count \n\n" + "Mean:" + str(round(np.mean(doc_lens))) + "   Median:" + str(round(np.median(doc_lens))) + "   Stdev:" + str(round(np.std(doc_lens))) + "   1%ile:" + str(round(np.quantile(doc_lens, q=0.01))) + "   99%ile:" + str(round(np.quantile(doc_lens, q=0.99))))
    plt.tick_params(size=20)
    plt.xticks(np.linspace(0,600,9))
    plt.title('Distribution of Document Word Counts', fontdict=dict(size=18))
    fig.tight_layout()
    flike = io.BytesIO()
    fig.savefig(flike)
    b64 = base64.b64encode(flike.getvalue()).decode()
    return b64
# Sentence Coloring of N Sentences
def topics_per_document(model, corpus, start=0, end=1):
    corpus_sel = corpus[start:end]
    dominant_topics = []
    topic_percentages = []
    for i, corp in enumerate(corpus_sel):
        topic_percs, wordid_topics, wordid_phivalues = model[corp]
        dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append((i, dominant_topic))
        topic_percentages.append(topic_percs)
    return(dominant_topics, topic_percentages)

def distrib_dominant(dominant_topics,lda_model,num_topics):
    # Distribution of Dominant Topics in Each Document
    df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
    dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
    df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()
    a=[]
    for i in range(0, num_topics):
        a.append(i)
    adf = pd.DataFrame(a, columns=['NumofTopics'])
    # Top 3 Keywords for each Topic
    topic_top3words = [(i, topic) for i, topics in lda_model.show_topics(num_topics=num_topics,formatted=False) 
                                    for j, (topic, wt) in enumerate(topics) if j < 10]
    df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
    df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
    df_top3words.reset_index(level=0,inplace=True)
    # Topic Distribution by Dominant Topics
    dominant_fig, ax1= plt.subplots(1, figsize=(25, 10),dpi=300, sharey=True)
    ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='firebrick')
    ax1.set_xticks(range(adf.NumofTopics.unique().__len__()))
    tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x+1)+ '\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0])
    ax1.xaxis.set_major_formatter(tick_formatter)
    ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=30))
    ax1.set_ylabel('Number of Documents',fontdict=dict(size=30))
    ax1.set_ylim(0, 1000)
    flike = io.BytesIO()
    dominant_fig.tight_layout()
    dominant_fig.savefig(flike)
    dom = base64.b64encode(flike.getvalue()).decode()
    return dom

def weightage_topic(topic_percentages,lda_model,num_topics):
    # Distribution of Dominant Topics in Each Document
    topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
    weightage_by_doc = round(topic_weightage_by_doc.sum(),2)
    df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()
    # Top 3 Keywords for each Topic
    topic_top3words = [(i, topic) for i, topics in lda_model.show_topics(num_topics=num_topics,formatted=False) 
                                    for j, (topic, wt) in enumerate(topics) if j < 10]
    df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
    df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
    df_top3words.reset_index(level=0,inplace=True)
    # Topic Distribution by Topic Weights
    weightage_fig,ax2 = plt.subplots(1, figsize=(25, 10),dpi=300, sharey=True)
    ax2.bar(x='index', height='count', data=df_topic_weightage_by_doc, width=.5, color='steelblue')
    ax2.set_xticks(range(df_topic_weightage_by_doc.index.unique().__len__()))
    tick_formatter = FuncFormatter(lambda x, pos: '\nTopic ' + str(x+1)+ '\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0])
    ax2.xaxis.set_major_formatter(tick_formatter)
    ax2.set_title('Number of Documents by Topic Weightage', fontdict=dict(size=30))
    ax2.set_ylabel('Number of Documents', fontdict=dict(size=30))
    ax2.set_ylim(0, 1000)
    weightage_fig.tight_layout()
    flike = io.BytesIO()
    weightage_fig.savefig(flike)
    weight = base64.b64encode(flike.getvalue()).decode()
    return weight

def word_cloud_gen(raw_data):
    def expand_contractions(inputs):
        expanded = []
        for sent in inputs:
            text_out = []
            for word in sent.split():
                text_out.append(contractions.fix(word))  
                expanded_text = ' '.join(text_out)
            expanded.append(expanded_text)
        return expanded
    stop_words = STOPWORDS
    stop_words = STOPWORDS.union(set(['really','especially','actually','feel','like','lot','studie']))
    sw_list = {'cannot','very','much','to','no','anything','own','just','everything','well','no','yes','not'}
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    def listing(inputs):
        expanded = []
        for sent in inputs:
            expanded_text = ' '.join(sent)
            expanded.append(expanded_text)
        return expanded
    stop_words = stop_words.difference(sw_list)
    data_test = expand_contractions(raw_data)
    data_words_test = list(sent_to_words(data_test)) #apply tokenization
    data_words_nostops_test = remove_stopwords(data_words_test)
    #forming bigrams and trigrams
    bigram = gensim.models.Phrases(data_words_nostops_test, min_count=5, threshold=0.5) 
    trigram = gensim.models.Phrases(bigram[data_words_nostops_test], threshold=0.5)  
    quadgram = gensim.models.Phrases(trigram[data_words_nostops_test],threshold=0.5)  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    quadram_mod = gensim.models.phrases.Phraser(quadgram)
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]
    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]
    def make_quadrams(texts):
        return [quadram_mod[trigram_mod[bigram_mod[doc]]] for doc in texts]
    #application of n-grams
    data_words_bigrams_test = make_bigrams(data_words_nostops_test)
    data_words_trigrams_test = make_trigrams(data_words_bigrams_test)
    data_words_quadrams_test = make_quadrams(data_words_trigrams_test)
    a = list(sent_to_words(data_words_quadrams_test)) 
    #improve_stop_words_test = remove_stopwords_improve(data_words_quadrams_test)
    improve_stop_words = listing(a)
    stop_words_improve = STOPWORDS
    stop_words_improve = STOPWORDS.union(set(['yes_not','yes_no','need_to','able_to','time_to','hard_to']))
    wordcloud = WordCloud(stopwords = stop_words_improve, width=1600,height=800,background_color='white').generate((str(improve_stop_words)))
    # create a figure
    word_fig, ax = plt.subplots(1,1, figsize = (5,3), dpi=300)
    # add interpolation = bilinear to smooth things out
    plt.imshow(wordcloud, interpolation='bilinear')
    # and remove the axis
    plt.axis("off")
    plt.tight_layout()
    flike = io.BytesIO()
    word_fig.savefig(flike)
    word = base64.b64encode(flike.getvalue()).decode()
    return word


def topic_list(ldamodel,num_topics):
  topics = ldamodel.show_topics(formatted=False, num_words=10,num_topics=num_topics, log=False)
  topic_summaries = {}
  for topic in topics:
      topic_index = topic[0]
      topic_word_weights = topic[1]
      topic_summaries[topic_index] = ' + '.join(
          f'{weight:.3f} * {word}' for word, weight in topic_word_weights[:10])
  topic = []
  for topic_index, topic_summary in topic_summaries.items():
    yes = "Topic " + str(topic_index +1) + ":   " + str(topic_summary)
    topic.append(yes)
  return topic

def word_count_graph(ldamodel,num_topics,improve_stop_words,y,x):
    topics = ldamodel.show_topics(num_topics=num_topics,formatted=False)
    data_flat = [w for w_list in improve_stop_words for w in w_list]
    counter = Counter(data_flat)
    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i , weight, counter[word]])
    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        
    # Plot Word Count and Weights of Topic Keywords
    word_count_fig, axes = plt.subplots(y, x, figsize=(25,20), sharey=True, dpi=300)
    cols = [color for name, color in mcolors.XKCD_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.7, label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(0, 0.090); ax.set_ylim(0, 100)
        ax.set_title('Topic: ' + str(i+1), color=cols[i], fontsize=10)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
        ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')
    word_count_fig.tight_layout(w_pad=2)    
    word_count_fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05) 
    flike = io.BytesIO()
    word_count_fig.savefig(flike)
    word_count = base64.b64encode(flike.getvalue()).decode()
    return word_count 


#def correct_word_spelling(inputs):
   # text_out = []
    #for word in inputs.split():
      #  word = Word(word)
       # text_out.append(word.correct())  
       # expanded_text = ' '.join(text_out)
   # return expanded_text
#DJANGO WEBSITE

def index(request):
    return render(request, 'landing.html')
 
def loginPage(request):
    if request.user.is_authenticated and request.user.is_admin:
        return redirect('student_list')
    else:
        if request.method == 'POST':
            email = request.POST.get('username')
            password = request.POST.get('password')
            user = authenticate(request, email=email, password=password)
            if user is not None:
                login(request, user)
                users = request.user
                if user.is_authenticated and users.is_admin:
                    users.last_login = datetime.today()
                    users.save()
                    return redirect('student_list')
                else:
                    messages.error (request,'You have entered an invalid email or password.')
                    return render(request, 'login.html')
            else:
                    messages.error (request,'You have entered an invalid email or password.')
                    return render(request, 'login.html')
    return render(request, 'login.html')
 
def logout_view(request):
    logout(request)
    return redirect('login')
 
def sign_upPage(request):
    return render(request, 'sign_up.html')
 
def student_navbar(request):
    return render(request, 'student/student_navbar.html')
 
def answer_summary(request):
    return render(request, 'student/answer_summary.html')
 
def start_survey(request):
    context={}
    date2day = date.today() #datetime = date + time
    context['date2day'] = date2day
    return render(request, 'student/start_survey.html',context)
 
def survey_question(request):
    context ={}
    date2day = date.today() #datetime = date + time
    context['date2day'] = date2day
    if request.POST:
        form = AnswerForm(request.POST)
        if form.is_valid():
            form.save()
            #q1 = form.cleaned_data.get('question1')
            #q2 = form.cleaned_data.get('question2')
            #q3 = form.cleaned_data.get('question3')
            #q4 = form.cleaned_data.get('question4')
            #q5 = form.cleaned_data.get('question5')
            #cq1 = correct_word_spelling(q1)
            #cq2 = correct_word_spelling(q2)
            #cq3 = correct_word_spelling(q3)
            #cq4 = correct_word_spelling(q4)
            #cq5 = correct_word_spelling(q5)
           # x.question1 = cq1
            #.question2 = cq2
            #x.question3 = cq3
            #x.question4 = cq4
           # x.question5 = cq5
            #x.save()
            return redirect('thankyou')
        else:
            context['register'] = form
    else:
        form = AnswerForm()
        context['register'] = form
    return render(request, 'student/survey_question.html', context)
 
def thankyou(request):
    return render(request, 'student/thankyou.html')

def admin_navbar(request):
    return render(request, 'admin/admin_navbar.html')

def evaluation(request):
    context={}
    param = 'Empty'
    date2day = date.today() #datetime = date + time
    #day = date.weekday() 
    day2day = datetime.today().weekday()
    if (request.method == 'POST'):
        csvFile = request.FILES.get('file')
        if not csvFile.name.endswith('.csv'):
            messages.error(request, 'Please only upload csv file')
        else:
            param = 'CSV'
            df_test = pd.read_csv(StringIO(csvFile.read().decode('utf-8')), delimiter=',')
            df_test['q1']=df_test['question1'].astype(str) #convert type to string
            df_test['q1']=df_test['q1'].apply(lambda x: x.lower()) #all lowercase
            df_test['q2']=df_test['question2'].astype(str) #convert type to string
            df_test['q2']=df_test['q2'].apply(lambda x: x.lower()) #all lowercase
            df_test['q3']=df_test['question3'].astype(str) #convert type to string
            df_test['q3']=df_test['q3'].apply(lambda x: x.lower()) #all lowercase
            df_test['q4']=df_test['question4'].astype(str) #convert type to string
            df_test['q4']=df_test['q4'].apply(lambda x: x.lower()) #all lowercase
            df_test['q5']=df_test['question5'].astype(str) #convert type to string
            df_test['q5']=df_test['q5'].apply(lambda x: x.lower()) #all lowercase

            raw_data_test1 = df_test.q1.values.tolist() #<--covert to list
            raw_data_test2 = df_test.q2.values.tolist() #<--covert to list
            raw_data_test3 = df_test.q3.values.tolist() #<--covert to list
            raw_data_test4 = df_test.q4.values.tolist() #<--covert to list
            raw_data_test5 = df_test.q5.values.tolist() #<--covert to list

        def cleaning1(raw_data_test):
            #temp_file = datapath('D:\capstone\capstone\capswebsite\model\lda_model1')
            #temp_file = datapath(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static\model\lda_model1'))
            #lda = models.ldamodel.LdaModel.load(temp_file)
            #dicts = corpora.Dictionary.load('D:\capstone\capstone\capswebsite\model\lda_model1.id2word')
            #id2word = corpora.Dictionary.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static\model\lda_model1.id2word'))
            lda = models.ldamodel.LdaModel.load('lda_model1')
            x1= lda.print_topics()
            id2word = corpora.Dictionary.load('lda_model1.id2word')
            def expand_contractions(inputs):
                contractions.add('profs', 'professors')
                contractions.add('professoressors', 'professors')
                contractions.add('prof', 'professor')
                contractions.add('f2f','face to face')
                contractions.add('ok','okay')
                contractions.add('papa','father')
                contractions.add('mom','mother')
                contractions.add('assign','assignment')
                contractions.add('acads','academics')
                contractions.add('acad','academic')
                contractions.add('distraction','distractions')
                expanded = []
                for sent in inputs:
                    text_out = []
                    for word in sent.split():
                        text_out.append(contractions.fix(word))  
                        expanded_text = ' '.join(text_out)
                    expanded.append(expanded_text)
                return expanded
            stop_words = STOPWORDS
            stop_words = STOPWORDS.union(set(['know','way','cause','specially','especially','create','come','make','become','like',
                                  'currently','really','bite','feel','add']))
            sw_list = {'cannot','very','much','lot','to','no','anything','own','just','everything','no','yes','not'}
            stop_words = stop_words.difference(sw_list)
            def remove_stopwords(texts):
                return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
            stop_words_improve = STOPWORDS
            stop_words_improve = STOPWORDS.union(set(['to_point','want_to','difficult_to','hard_to','able_to','tend_to','not_able_to','unable_to']))
            #def remove_stopwords_improve(texts):
                #return [[word for word in simple_preprocess(str(doc)) if word not in stop_words_improve] for doc in texts]
            #application of contraction,tokenization,lemmatization,stop_words
            data_test = expand_contractions(raw_data_test)
            data_words_test = list(sent_to_words(data_test)) #apply tokenization
            data_lemmatized_test = lemmatization(data_words_test)
            data_words_nostops_test = remove_stopwords(data_lemmatized_test)
            #forming bigrams and trigrams
            bigram = gensim.models.Phrases(data_words_nostops_test, min_count=2, threshold=5) 
            trigram = gensim.models.Phrases(bigram[data_words_nostops_test], threshold=5)  
            quadgram = gensim.models.Phrases(trigram[data_words_nostops_test],threshold=5)  
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            trigram_mod = gensim.models.phrases.Phraser(trigram)
            quadram_mod = gensim.models.phrases.Phraser(quadgram)
            def make_bigrams(texts):
                return [bigram_mod[doc] for doc in texts]
            def make_trigrams(texts):
                return [trigram_mod[bigram_mod[doc]] for doc in texts]
            def make_quadrams(texts):
                return [quadram_mod[trigram_mod[bigram_mod[doc]]] for doc in texts]
            #application of n-grams
            data_words_bigrams_test = make_bigrams(data_words_nostops_test)
            data_words_trigrams_test = make_trigrams(data_words_bigrams_test)
            data_words_quadrams_test = make_quadrams(data_words_trigrams_test)
            #improve_stop_words_test = remove_stopwords_improve(data_words_quadrams_test)
            improve_stop_words_test = data_words_quadrams_test
            
            texts_test = improve_stop_words_test
            corpus_test = [id2word.doc2bow(text) for text in texts_test]
            tfidf = TfidfModel(corpus_test,id2word=id2word)
            low_value = 0.03
            words = []
            words_missing_in_tfidf = [] #reinitialize to be safe. You can skip this
            for i in range(0, len(corpus_test)):
                bow = corpus_test[i]
                low_value_words = []
                tfidf_ids = [id for id, value in tfidf[bow]]
                bow_ids =  [id for id, value in bow]
                low_value_words  = [id for id, value in tfidf[bow] if value < low_value]
                drops =  low_value_words+words_missing_in_tfidf
                for item in drops:
                    words.append(id2word[item])
                words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids] #The words with tf-idf score 0 will be missing
                new_bow  =  [b for b in bow if b[0] not in low_value_words and b[0]  not in words_missing_in_tfidf]
                corpus_test[i] =  new_bow
            unseen_doc = corpus_test
            vector = lda[unseen_doc]
            lda.update(corpus_test)
            vector = lda[unseen_doc]
            return improve_stop_words_test,texts_test,corpus_test,lda

        def cleaning2(raw_data_test):
            #temp_file = datapath('D:\capstone\capstone\capswebsite\model\lda_model1')
            #temp_file = datapath(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static\model\lda_model2'))
            #lda = models.ldamodel.LdaModel.load(temp_file)
            #id2word = corpora.Dictionary.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static\model\lda_model2.id2word'))
            lda = models.ldamodel.LdaModel.load('lda_model2')
            x1= lda.print_topics()
            id2word = corpora.Dictionary.load('lda_model2.id2word')
            def expand_contractions(inputs):
                contractions.add('profs', 'professors')
                contractions.add('prof', 'professor')
                contractions.add('f2f','face to face')
                contractions.add('ok','okay')
                contractions.add('exams','examinations')
                contractions.add('exam','examination')
                contractions.add('sem','semester')
                contractions.add('professoressors','professors')
                contractions.add('final','finals')
                contractions.add('acads','academics')
                contractions.add('acad','academics')
                contractions.add('assigns','assignments')
                contractions.add('assign','assignments')
                expanded = []
                for sent in inputs:
                    text_out = []
                    for word in sent.split():
                        text_out.append(contractions.fix(word))  
                        expanded_text = ' '.join(text_out)
                    expanded.append(expanded_text)
                return expanded
            stop_words = STOPWORDS
            stop_words = STOPWORDS.union(set(['react','date','specially','think','far','honestly','foo','come','ask','look','past','end','nott',
                                  'pretty','gon','si','thing','slightly','lately','anymore','especially','haha','cause','guess','usually','like',
                                  'know','currently','actually','let','felt','past','use','bite','try']))
            sw_list = {'cannot','very','much','lot','to','no','anything','own','just','everything','well','no','yes','not'}
            stop_words = stop_words.difference(sw_list)
            def remove_stopwords(texts):
                return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
            stop_words_improve = STOPWORDS
            stop_words_improve = STOPWORDS.union(set(['just_want','want_to','able_to']))
            #def remove_stopwords_improve(texts):
                #return [[word for word in simple_preprocess(str(doc)) if word not in stop_words_improve] for doc in texts]
            #application of contraction,tokenization,lemmatization,stop_words
            data_test = expand_contractions(raw_data_test)
            data_words_test = list(sent_to_words(data_test)) #apply tokenization
            data_lemmatized_test = lemmatization(data_words_test)
            data_words_nostops_test = remove_stopwords(data_lemmatized_test)
            #forming bigrams and trigrams
            bigram = gensim.models.Phrases(data_words_nostops_test, min_count=2, threshold=5) 
            trigram = gensim.models.Phrases(bigram[data_words_nostops_test], threshold=5)  
            quadgram = gensim.models.Phrases(trigram[data_words_nostops_test],threshold=5)  
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            trigram_mod = gensim.models.phrases.Phraser(trigram)
            quadram_mod = gensim.models.phrases.Phraser(quadgram)
            def make_bigrams(texts):
                return [bigram_mod[doc] for doc in texts]
            def make_trigrams(texts):
                return [trigram_mod[bigram_mod[doc]] for doc in texts]
            def make_quadrams(texts):
                return [quadram_mod[trigram_mod[bigram_mod[doc]]] for doc in texts]
            #application of n-grams
            data_words_bigrams_test = make_bigrams(data_words_nostops_test)
            data_words_trigrams_test = make_trigrams(data_words_bigrams_test)
            data_words_quadrams_test = make_quadrams(data_words_trigrams_test)
            #improve_stop_words_test = remove_stopwords_improve(data_words_quadrams_test)
            improve_stop_words_test = data_words_quadrams_test
            
            texts_test = improve_stop_words_test
            corpus_test = [id2word.doc2bow(text) for text in texts_test]
            tfidf = TfidfModel(corpus_test,id2word=id2word)
            low_value = 0.05
            words = []
            words_missing_in_tfidf = [] #reinitialize to be safe. You can skip this
            for i in range(0, len(corpus_test)):
                bow = corpus_test[i]
                low_value_words = []
                tfidf_ids = [id for id, value in tfidf[bow]]
                bow_ids =  [id for id, value in bow]
                low_value_words  = [id for id, value in tfidf[bow] if value < low_value]
                drops =  low_value_words+words_missing_in_tfidf
                for item in drops:
                    words.append(id2word[item])
                words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids] #The words with tf-idf score 0 will be missing

                new_bow  =  [b for b in bow if b[0] not in low_value_words and b[0]  not in words_missing_in_tfidf]
                corpus_test[i] =  new_bow
            unseen_doc = corpus_test
            vector = lda[unseen_doc]
            lda.update(corpus_test)
            vector = lda[unseen_doc]
            return improve_stop_words_test,texts_test,corpus_test,lda


        def cleaning3(raw_data_test):
            #temp_file = datapath('D:\capstone\capstone\capswebsite\model\lda_model1')
            #temp_file = datapath(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static\model\lda_model3'))
            #lda = models.ldamodel.LdaModel.load(temp_file)
            #id2word = corpora.Dictionary.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static\model\lda_model3.id2word'))
            lda = models.ldamodel.LdaModel.load('lda_model3')
            x1= lda.print_topics()
            id2word = corpora.Dictionary.load('lda_model3.id2word')
            def expand_contractions(inputs):
                contractions.add('profs', 'professors')
                contractions.add('prof', 'professor')
                contractions.add('f2f','face to face')
                contractions.add('ok','okay')
                contractions.add('hrs','hours')
                expanded = []
                for sent in inputs:
                    text_out = []
                    for word in sent.split():
                        text_out.append(contractions.fix(word))  
                        expanded_text = ' '.join(text_out)
                    expanded.append(expanded_text)
                return expanded
            def expand_contractions_improve(inputs):
                contractions.add('mentally_physically', 'physically_mentally')
                contractions.add('mentally_physically_fatigue', 'physically_mentally_fatigue')
                expanded = []
                for sent in inputs:
                    text_out = []
                    for word in sent:
                        text_out.append(contractions.fix(word))  
                    expanded.append(text_out)
                return expanded
            stop_words = STOPWORDS
            stop_words = STOPWORDS.union(set(['actually', 'bite', 'wear', 'sure', 'things', 'definitely']))
            sw_list = {'cannot','not','do','can','should','would','very','much','too','lot','really','to','sometimes','of','does','no'}
            stop_words = stop_words.difference(sw_list)
            def remove_stopwords(texts):
                return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
            stop_words_improve = STOPWORDS
            stop_words_improve = STOPWORDS.union(set(['physically', 'mentally', 'physically_mentally', 'mentally_physically']))
            #def remove_stopwords_improve(texts):
                #return [[word for word in simple_preprocess(str(doc)) if word not in stop_words_improve] for doc in texts]
            #application of contraction,tokenization,lemmatization,stop_words
            data_test = expand_contractions(raw_data_test)
            data_words_test = list(sent_to_words(data_test)) #apply tokenization
            data_lemmatized_test = lemmatization(data_words_test)
            data_words_nostops_test = remove_stopwords(data_lemmatized_test)
            #forming bigrams and trigrams
            bigram = gensim.models.Phrases(data_words_nostops_test, min_count=2, threshold=5) 
            trigram = gensim.models.Phrases(bigram[data_words_nostops_test], threshold=5)  
            quadgram = gensim.models.Phrases(trigram[data_words_nostops_test],threshold=5)  
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            trigram_mod = gensim.models.phrases.Phraser(trigram)
            quadram_mod = gensim.models.phrases.Phraser(quadgram)
            def make_bigrams(texts):
                return [bigram_mod[doc] for doc in texts]
            def make_trigrams(texts):
                return [trigram_mod[bigram_mod[doc]] for doc in texts]
            def make_quadrams(texts):
                return [quadram_mod[trigram_mod[bigram_mod[doc]]] for doc in texts]
            #application of n-grams
            data_words_bigrams_test = make_bigrams(data_words_nostops_test)
            data_words_trigrams_test = make_trigrams(data_words_bigrams_test)
            data_words_quadrams_test = make_quadrams(data_words_trigrams_test)
            #improve_stop_words_test = remove_stopwords_improve(data_words_quadrams_test)
            #improve_stop_words_test = data_words_quadrams_test
            improve_stop_words_test= expand_contractions_improve(data_words_quadrams_test)
            clean = []
            
            texts_test = improve_stop_words_test
            corpus_test = [id2word.doc2bow(text) for text in texts_test]
            tfidf = TfidfModel(corpus_test,id2word=id2word)
            low_value = 0.05
            words = []
            words_missing_in_tfidf = [] #reinitialize to be safe. You can skip this
            for i in range(0, len(corpus_test)):
                bow = corpus_test[i]
                low_value_words = []
                tfidf_ids = [id for id, value in tfidf[bow]]
                bow_ids =  [id for id, value in bow]
                low_value_words  = [id for id, value in tfidf[bow] if value < low_value]
                drops =  low_value_words+words_missing_in_tfidf
                for item in drops:
                    words.append(id2word[item])
                words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids] #The words with tf-idf score 0 will be missing

                new_bow  =  [b for b in bow if b[0] not in low_value_words and b[0]  not in words_missing_in_tfidf]
                corpus_test[i] =  new_bow
            unseen_doc = corpus_test
            vector = lda[unseen_doc]
            lda.update(corpus_test)
            vector = lda[unseen_doc]
            return improve_stop_words_test,texts_test,corpus_test,lda

        def cleaning4(raw_data_test):
            #temp_file = datapath('D:\capstone\capstone\capswebsite\model\lda_model1')
            #temp_file = datapath(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static\model\lda_model4'))
            #lda = models.ldamodel.LdaModel.load(temp_file)
            #id2word = corpora.Dictionary.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static\model\lda_model4.id2word'))
            lda = models.ldamodel.LdaModel.load('lda_model4')
            x1= lda.print_topics()
            id2word = corpora.Dictionary.load('lda_model4.id2word')
            def expand_contractions(inputs):
                contractions.add('profs', 'professors')
                contractions.add('prof', 'professor')
                contractions.add('f2f','face to face')
                contractions.add('ok','okay')
                contractions.add('hrs','hours')
                expanded = []
                for sent in inputs:
                    text_out = []
                    for word in sent.split():
                        text_out.append(contractions.fix(word))  
                        expanded_text = ' '.join(text_out)
                    expanded.append(expanded_text)
                return expanded
            def expand_contractions_improve(inputs):
                contractions.add('mentally_physically', 'physically_mentally')
                contractions.add('mentally_physically_fatigue', 'physically_mentally_fatigue')
                expanded = []
                for sent in inputs:
                    text_out = []
                    for word in sent:
                        text_out.append(contractions.fix(word))  
                    expanded.append(text_out)
                return expanded
            stop_words = STOPWORDS
            stop_words = STOPWORDS.union(set(['yes', 'meet', 'come', 'bite']))
            sw_list = {'cannot','not','do','can','should','would','very','much','too','lot','really','to','sometimes','of','does','no'}
            stop_words = stop_words.difference(sw_list)
            def remove_stopwords(texts):
                return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
            stop_words_improve = STOPWORDS
            stop_words_improve = STOPWORDS.union(set(['face_to']))
            #def remove_stopwords_improve(texts):
                #return [[word for word in simple_preprocess(str(doc)) if word not in stop_words_improve] for doc in texts]
            #application of contraction,tokenization,lemmatization,stop_words
            data_test = expand_contractions(raw_data_test)
            data_words_test = list(sent_to_words(data_test)) #apply tokenization
            data_lemmatized_test = lemmatization(data_words_test)
            data_words_nostops_test = remove_stopwords(data_lemmatized_test)
            #forming bigrams and trigrams
            bigram = gensim.models.Phrases(data_words_nostops_test, min_count=2, threshold=5) 
            trigram = gensim.models.Phrases(bigram[data_words_nostops_test], threshold=5)  
            quadgram = gensim.models.Phrases(trigram[data_words_nostops_test],threshold=5)  
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            trigram_mod = gensim.models.phrases.Phraser(trigram)
            quadram_mod = gensim.models.phrases.Phraser(quadgram)
            def make_bigrams(texts):
                return [bigram_mod[doc] for doc in texts]
            def make_trigrams(texts):
                return [trigram_mod[bigram_mod[doc]] for doc in texts]
            def make_quadrams(texts):
                return [quadram_mod[trigram_mod[bigram_mod[doc]]] for doc in texts]
            #application of n-grams
            data_words_bigrams_test = make_bigrams(data_words_nostops_test)
            data_words_trigrams_test = make_trigrams(data_words_bigrams_test)
            data_words_quadrams_test = make_quadrams(data_words_trigrams_test)
            #improve_stop_words_test = remove_stopwords_improve(data_words_quadrams_test)
            #improve_stop_words_test = data_words_quadrams_test
            improve_stop_words_test= expand_contractions_improve(data_words_quadrams_test)
            
            texts_test = improve_stop_words_test
            corpus_test = [id2word.doc2bow(text) for text in texts_test]
            tfidf = TfidfModel(corpus_test,id2word=id2word)
            low_value = 0.05
            words = []
            words_missing_in_tfidf = [] #reinitialize to be safe. You can skip this
            for i in range(0, len(corpus_test)):
                bow = corpus_test[i]
                low_value_words = []
                tfidf_ids = [id for id, value in tfidf[bow]]
                bow_ids =  [id for id, value in bow]
                low_value_words  = [id for id, value in tfidf[bow] if value < low_value]
                drops =  low_value_words+words_missing_in_tfidf
                for item in drops:
                    words.append(id2word[item])
                words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids] #The words with tf-idf score 0 will be missing

                new_bow  =  [b for b in bow if b[0] not in low_value_words and b[0]  not in words_missing_in_tfidf]
                corpus_test[i] =  new_bow
            unseen_doc = corpus_test
            vector = lda[unseen_doc]
            lda.update(corpus_test)
            vector = lda[unseen_doc]
            return improve_stop_words_test,texts_test,corpus_test,lda            

        def cleaning5(raw_data_test):
            #temp_file = datapath('D:\capstone\capstone\capswebsite\model\lda_model1')
            #temp_file = datapath(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static\model\lda_model5'))
            #lda = models.ldamodel.LdaModel.load(temp_file)
            #id2word = corpora.Dictionary.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static\model\lda_model5.id2word'))
            lda = models.ldamodel.LdaModel.load('lda_model5')
            x1= lda.print_topics()
            id2word = corpora.Dictionary.load('lda_model5.id2word')
            def expand_contractions(inputs):
                contractions.add('profs', 'professors')
                contractions.add('professoressors', 'professors')
                contractions.add('prof', 'professor')
                contractions.add('f2f','face to face')
                contractions.add('ok','okay')
                contractions.add('papa','father')
                contractions.add('mom','mother')
                contractions.add('assign','assignment')
                contractions.add('acads','academics')
                contractions.add('acad','academics')
                expanded = []
                for sent in inputs:
                    text_out = []
                    for word in sent.split():
                        text_out.append(contractions.fix(word))  
                        expanded_text = ' '.join(text_out)
                    expanded.append(expanded_text)
                return expanded
            stop_words = STOPWORDS
            stop_words = STOPWORDS.union(set(['yes','tree','know','way','cause','specially','especially','create','come','ung','make','become','like','also','able',
                                  'currently','really','have','lot', 'belong', 'bond', 'people', 'person', 'aside', 'ypur', 'house', 'anymore', 
                                  'mean', 'use', 'happen', 'add', 'certain', 'feel', 'honestly', 'rise', 'sisters', 'ny', 'term',
                                  'think', 'felt', 'long', 'lessen', 'small', 'real', 'daughter', 'reason']))
            sw_list = {'cannot','not','do','can','should','would','very','much','too','lot','alot','really','to','sometimes','of','does','no', 'of'}
            stop_words = stop_words.difference(sw_list)
            def remove_stopwords(texts):
                return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
            stop_words_improve = STOPWORDS
            stop_words_improve = STOPWORDS.union(set(['do_not', 'of ways', 'instead of']))
            #def remove_stopwords_improve(texts):
                #return [[word for word in simple_preprocess(str(doc)) if word not in stop_words_improve] for doc in texts]
            #application of contraction,tokenization,lemmatization,stop_words
            data_test = expand_contractions(raw_data_test)
            data_words_test = list(sent_to_words(data_test)) #apply tokenization
            data_lemmatized_test = lemmatization(data_words_test)
            data_words_nostops_test = remove_stopwords(data_lemmatized_test)
            #forming bigrams and trigrams
            bigram = gensim.models.Phrases(data_words_nostops_test, min_count=2, threshold=5) 
            trigram = gensim.models.Phrases(bigram[data_words_nostops_test], threshold=5)  
            quadgram = gensim.models.Phrases(trigram[data_words_nostops_test],threshold=5)  
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            trigram_mod = gensim.models.phrases.Phraser(trigram)
            quadram_mod = gensim.models.phrases.Phraser(quadgram)
            def make_bigrams(texts):
                return [bigram_mod[doc] for doc in texts]
            def make_trigrams(texts):
                return [trigram_mod[bigram_mod[doc]] for doc in texts]
            def make_quadrams(texts):
                return [quadram_mod[trigram_mod[bigram_mod[doc]]] for doc in texts]
            #application of n-grams
            data_words_bigrams_test = make_bigrams(data_words_nostops_test)
            data_words_trigrams_test = make_trigrams(data_words_bigrams_test)
            data_words_quadrams_test = make_quadrams(data_words_trigrams_test)
            #improve_stop_words_test = remove_stopwords_improve(data_words_quadrams_test)
            improve_stop_words_test = data_words_quadrams_test
            
            texts_test = improve_stop_words_test
            corpus_test = [id2word.doc2bow(text) for text in texts_test]
            tfidf = TfidfModel(corpus_test,id2word=id2word)
            low_value = 0.02
            words = []
            words_missing_in_tfidf = [] #reinitialize to be safe. You can skip this
            for i in range(0, len(corpus_test)):
                bow = corpus_test[i]
                low_value_words = []
                tfidf_ids = [id for id, value in tfidf[bow]]
                bow_ids =  [id for id, value in bow]
                low_value_words  = [id for id, value in tfidf[bow] if value < low_value]
                drops =  low_value_words+words_missing_in_tfidf
                for item in drops:
                    words.append(id2word[item])
                words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids] #The words with tf-idf score 0 will be missing

                new_bow  =  [b for b in bow if b[0] not in low_value_words and b[0]  not in words_missing_in_tfidf]
                corpus_test[i] =  new_bow
            unseen_doc = corpus_test
            vector = lda[unseen_doc]
            lda.update(corpus_test)
            vector = lda[unseen_doc]
            return improve_stop_words_test,texts_test,corpus_test,lda
    
        #question1   
        improve_stop_words_test1,texts_test1,corpus_test1,lda_model1 = cleaning1(raw_data_test1)
        #dominant and weightage
        dominant_topics1, topic_percentages1 = topics_per_document(model=lda_model1, corpus=corpus_test1, end=-1)
        dom_plot1 = distrib_dominant(dominant_topics1,lda_model1,20)
        weightage_plot1 = weightage_topic(topic_percentages1,lda_model1,20)
        word_list1 = topic_list(lda_model1,20)
        count_graph1 = word_count_graph(lda_model1,20,improve_stop_words_test1,4,5)

        #question2
        improve_stop_words_test2,texts_test2,corpus_test2,lda_model2= cleaning2(raw_data_test2)
        #dominant and weightage
        dominant_topics2, topic_percentages2 = topics_per_document(model=lda_model2, corpus=corpus_test2, end=-1)
        dom_plot2 = distrib_dominant(dominant_topics2,lda_model2,15)
        weightage_plot2 = weightage_topic(topic_percentages2,lda_model2,15)
        word_list2 = topic_list(lda_model2,20)
        count_graph2 = word_count_graph(lda_model2,15,improve_stop_words_test2,3,5)

        #question3
        improve_stop_words_test3,texts_test3,corpus_test3,lda_model3= cleaning3(raw_data_test3)
        #dominant and weightage
        dominant_topics3, topic_percentages3 = topics_per_document(model=lda_model3, corpus=corpus_test3, end=-1)
        dom_plot3 = distrib_dominant(dominant_topics3,lda_model3,20)
        weightage_plot3 = weightage_topic(topic_percentages3,lda_model3,20)
        word_list3 = topic_list(lda_model3,25)
        count_graph3 = word_count_graph(lda_model3,25,improve_stop_words_test3,4,5)

        #question4
        improve_stop_words_test4,texts_test4,corpus_test4,lda_model4= cleaning4(raw_data_test4)
        #dominant and weightage
        dominant_topics4, topic_percentages4 = topics_per_document(model=lda_model4, corpus=corpus_test4, end=-1)
        dom_plot4 = distrib_dominant(dominant_topics4,lda_model4,20)
        weightage_plot4 = weightage_topic(topic_percentages4,lda_model4,20)
        word_list4 = topic_list(lda_model4,30)
        count_graph4 = word_count_graph(lda_model4,30,improve_stop_words_test4,4,5)

        #question5
        improve_stop_words_test5,texts_test5,corpus_test5,lda_model5= cleaning5(raw_data_test5)
        #dominant and weightage
        dominant_topics5, topic_percentages5 = topics_per_document(model=lda_model5, corpus=corpus_test5, end=-1)
        dom_plot5 = distrib_dominant(dominant_topics5,lda_model5,20)
        weightage_plot5 = weightage_topic(topic_percentages5,lda_model5,20)
        word_list5 = topic_list(lda_model5,20)
        count_graph5 = word_count_graph(lda_model5,20,improve_stop_words_test5,4,5)


        #q1
        context['dominant1'] = dom_plot1
        context['weightage1'] = weightage_plot1
        context['word_list1'] = word_list1
        context['count_graph1'] = count_graph1
        #q2
        context['dominant2'] = dom_plot2
        context['weightage2'] = weightage_plot2
        context['word_list2'] = word_list2
        context['count_graph2'] = count_graph2
        #q3
        context['dominant3'] = dom_plot3
        context['weightage3'] = weightage_plot3
        context['word_list3'] = word_list3
        context['count_graph3'] = count_graph3
        #4
        context['dominant4'] = dom_plot4
        context['weightage4'] = weightage_plot4
        context['word_list4'] = word_list4
        context['count_graph4'] = count_graph4
        #5
        context['dominant5'] = dom_plot5
        context['weightage5'] = weightage_plot5
        context['word_list5'] = word_list5
        context['count_graph5'] = count_graph5
        
        context['param'] = param
        #pprint(lda.print_topics())
        #x1= lda_model1.print_topics()
        #context['x1'] = x1
        #x2= lda_model2.print_topics()
        #context['x2'] = x2
    context['param'] = param
    context['date2day'] = date2day
    context['day2day'] = day2day
    #context['day'] = day
    
    return render(request, 'admin/evaluation.html',context)

 
def student_list(request):
    context={}
    date2day = date.today() #datetime = date + time
    answers = Answers.objects.all()
    answers_count = answers.count()
    if (request.method == 'POST'):
        csvFile = request.FILES.get('file')
        if not csvFile.name.endswith('.csv'):
            messages.error(request, 'Please only upload csv file')
        else:
            dataset = StringIO(csvFile.read().decode('utf-8'))
            next(dataset)
            for row in csv.reader(dataset, delimiter=','):
                _, create = Answers.objects.update_or_create(
                    pk = row[0],
                    email=row[1],
                    firstName = row[2],
                    lastName = row[3],
                    numberID = row[4],
                    college_id = row[5],
                    course_id = row[6],
                    year = row[7],
                    block = row[8],
                    question1 = row[9],
                    question2 = row[10],
                    question3 = row[11],
                    question4 = row[12],
                    question5 = row[13],
                )
    searchthis_query = request.GET.get('searchthis')
    if searchthis_query != " " and searchthis_query is not None:
        answers = Answers.objects.filter(Q(numberID=searchthis_query)).distinct()
    context['date2day'] = date2day
    context['answers'] = answers
    context['answers_count'] = answers_count
    return render(request,'admin/student_list.html',context)
    
def student_answer(request, pk):
    context={}
    date2day = date.today() #datetime = date + time
    answers = Answers.objects.filter(id=pk)
    context = {'answers':answers}
    context['date2day'] = date2day
    return render(request, 'admin/student_answer.html',context)
 
def load_slot(request):
    collegeId = request.GET.get('college_Id')
    course = Course.objects.filter(college=collegeId)
    return render(request, 'student/dropdown_option.html', {'course': course})

def answers_csv(request): #csv bago toh
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename=MH_answers.csv'

    #Create a csv writer
    writer = csv.writer(response)

    #Designate the model
    answers = Answers.objects.all()

    #Add column headings to the csv file
    writer.writerow(['id','email', 'firstName', 'lastName', 'numberID', 'college', 'course', 'year', 'block', 'question1', 'question2', 'question3', 'question4', 'question5'])

    #Loop thru and output
    for answer in answers:
        writer.writerow([answer.pk,answer.email, answer.firstName, answer.lastName, answer.numberID, answer.college.id, answer.course.id, answer.year, answer.block, answer.question1, answer.question2, answer.question3, answer.question4, answer.question5])

    return response


def word_cloud_page(request):
    context={}
    date2day = date.today() #datetime = date + time
    answers = Answers.objects.all().values()
    ans_count = Answers.objects.all().count()
    print(ans_count)
    if ans_count == 0:
        param = "Empty"
        context['param'] = param
    else:
        param = "Not Empty"
        df = pd.DataFrame(answers)
        #print(df.head(10))
        df['q1']=df['question1'].astype(str) #convert type to string
        df['q1']=df['q1'].apply(lambda x: x.lower()) #all lowercase
        df['q2']=df['question2'].astype(str) #convert type to string
        df['q2']=df['q2'].apply(lambda x: x.lower()) #all lowercase
        df['q3']=df['question3'].astype(str) #convert type to string
        df['q3']=df['q3'].apply(lambda x: x.lower()) #all lowercase
        df['q4']=df['question4'].astype(str) #convert type to string
        df['q4']=df['q4'].apply(lambda x: x.lower()) #all lowercase
        df['q5']=df['question5'].astype(str) #convert type to string
        df['q5']=df['q5'].apply(lambda x: x.lower()) #all lowercase
        raw_data_test1 = df.q1.values.tolist() #<--covert to list
        raw_data_test2 = df.q2.values.tolist() #<--covert to list
        raw_data_test3 = df.q3.values.tolist() #<--covert to list
        raw_data_test4 = df.q4.values.tolist() #<--covert to list
        raw_data_test5 = df.q5.values.tolist() #<--covert to list
        word_cloud1 = word_cloud_gen(raw_data_test1)
        word_cloud2 = word_cloud_gen(raw_data_test2)
        word_cloud3 = word_cloud_gen(raw_data_test3)
        word_cloud4 = word_cloud_gen(raw_data_test4)
        word_cloud5 = word_cloud_gen(raw_data_test5)
        context['word_cloud1'] = word_cloud1
        context['word_cloud2'] = word_cloud2
        context['word_cloud3'] = word_cloud3
        context['word_cloud4'] = word_cloud4
        context['word_cloud5'] = word_cloud5
    context['param'] = param
    context['date2day'] = date2day
    return render(request, 'word_cloud.html', context)
    