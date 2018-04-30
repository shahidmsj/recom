import pandas as pd
import numpy as np
import seaborn as sns
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import warnings; warnings.simplefilter('ignore')
from nltk.stem.snowball import SnowballStemmer


md=pd.read_csv("/home/shahid/Downloads/movie-dataset/movies_metadata.csv")
link_small=pd.read_csv("/home/shahid/Downloads/movie-dataset/links_small.csv")

md['genres']=md['genres'].fillna('').apply(literal_eval).apply(
        lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

link_small=link_small[link_small['tmdbId'].notnull()]['tmdbId'].astype('int')


md=md.drop([19730, 29503, 35587])
md['id']=md['id'].astype('int')
smd=md[md['id'].isin(link_small)]
smd['tagline']=md['tagline'].fillna('')
smd['description']=smd['tagline']+smd['overview']
smd['description']=smd['description'].fillna('')

tf=TfidfVectorizer(input='word', stop_words='english')
tfidf_matrix=tf.fit_transform(smd['description'])


cosine_sim=linear_kernel(tfidf_matrix, tfidf_matrix)

smd=smd.reset_index()
titles=smd['title']
indices=pd.Series(smd.index, index=smd['title'])

#def get_recommendations(title):
#    idx=indices[title]
#    sim_score=list(enumerate(cosine_sim[idx]))
#    sim_score.sort(key=lambda x: x[1], reverse=True)
#    sim_score=sim_score[1:31]
#    movie_indices=[i[0] for i in sim_score]
#    return titles.iloc[movie_indices]


##########     metadata based movie recommender      #############

credit=pd.read_csv("/home/shahid/Downloads/movie-dataset/credits.csv")
keywords=pd.read_csv("/home/shahid/Downloads/movie-dataset/keywords.csv")

keywords['id'] = keywords['id'].astype('int')
credit['id'] = credit['id'].astype('int')

md['id'] = md['id'].astype('int')

md=md.merge(credit, on='id')
md=md.merge(keywords, on='id')

smd=md[md['id'].isin(link_small)]


smd['cast']=smd['cast'].apply(literal_eval)
smd['crew']=smd['crew'].apply(literal_eval)
smd['keywords']=smd['keywords'].apply(literal_eval)
smd['cast_size']=smd['cast'].apply(lambda x: len(x))
smd['crew_size']=smd['crew'].apply(lambda x: len(x))

def get_director(x):
    for i in x:    
        if i['job']=='Director':
            return i['name']
        else:
            return np.nan
    
smd['director']=smd['crew'].apply(get_director)

smd['cast']=smd['cast'].apply(
        lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    
smd['keywords']=smd['keywords'].apply(
        lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

smd['cast']=smd['cast'].apply(
        lambda x: [str.lower(i.replace(" ","")) for i in x])

smd['director']=smd['director'].astype('str').apply(
        lambda x: str.lower(x.replace(" ","")))

smd['director']=smd['director'].apply(lambda x: [x,x,x])


s=smd.apply(lambda x: pd.Series(x['keywords']), axis=1).stack(
        ).reset_index(level=1, drop=True)
s.name='keyword'

s=s.value_counts()

s=s[s>1]
stemmer=SnowballStemmer("english")

#
#def filter_keyword(x):
#    words=[]
#    for i in x:
#        words.append(i)
#    return words
#
#
#smd['keywords']=smd['keywords'].apply(filter_keyword)
smd['keywords']=smd['keywords'].apply(
        lambda x: [stemmer.stem(i) for i in x])

smd['keywords']=smd['keywords'].apply(
        lambda x: [str.lower(i.replace(" ","")) for i in x])

smd['soup']=smd['keywords']+smd['cast']+smd['genres']+smd['director']
smd['soup']=smd['soup'].apply(lambda x: ' '.join(x))
count=CountVectorizer(input='word', stop_words='english')
count_matrix=count.fit_transform(smd['soup'])
cos_sim=cosine_similarity(count_matrix, count_matrix)
smd=smd.reset_index()
titles=smd['title']
indices=pd.Series(smd.index, index=smd['title'])


def get_recommendations(title):
    idx=indices[title]
    sim_scores=list(enumerate(cos_sim[idx]))
    sim_scores=sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores=sim_scores[1:31]
    movie_indices=[i[0]for i in sim_scores]
    return titles.iloc[movie_indices]

print("Enter name : ")
a=input()
print(get_recommendations(a))



