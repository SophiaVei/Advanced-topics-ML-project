import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def generate_sentim_features(tweets,lex_uni, embeds):
    
    embedding_features = generate_average_embedding_features(tweets, embeds)
    unigram_lex_features = generate_lexicon_features(tweets, lex_uni, 1)
    
    return unigram_lex_features, embedding_features


def extract_ngram_features(tweets):
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=10, binary=True)
    return vectorizer.fit_transform(tweets).toarray(), vectorizer

    
def generate_average_embedding_features(tweets, word_vectors):
    all_scores = []
    c=0
    for tweet in tweets:
        tweet_scores = []
        words = extract_ngrams(tweet, 1)
        for word in words:
            tweet_scores.append(word_resource_score(word, word_vectors, is_embed=True))
        tweet_scores = np.array(tweet_scores)
        if len(words)==0:
            val = np.zeros(300)
        else:
            val = np.nanmean(np.array(tweet_scores), axis=0) #sum over all words, [300,]
            
            '''check how many instances are zeros'''
            val_sum = np.nansum(val)
            if val_sum==0:
                c+=1
        all_scores.append(val)
    #print(c)
    return np.array(all_scores)


def generate_embedding_features(tweets, word_vectors):
    '''create a list of arrays of embeddings for each tweet for all tweets'''
    all_scores = []
    c=0
    for tweet in tweets:
        tweet_scores = []
        words = extract_ngrams(tweet, 1)
        for word in words:
            tweet_scores.append(word_resource_score(word, word_vectors, is_embed=True))
        tweet_scores = np.array(tweet_scores)
        '''if len(words)==0:
            val = np.zeros(300)
        else:
            val = np.nanmean(np.array(tweet_scores), axis=0) #sum over all words, [300,]
            
            #check how many instances are zeros
            val_sum = np.nansum(val)
            if val_sum==0:
                c+=1'''
        all_scores.append(tweet_scores)
    print('len for all tweets',len(all_scores))      
    return all_scores
    
    


def generate_lexicon_features(tweets, lexicons, ngrams):
    '''tweets: list of tweet strings'''
    
    all_scores = []
    for tweet in tweets:  #for each tweet
        tweet_scores = []
        words = extract_ngrams(tweet, ngrams)
        for word in words: #for each word
            '''search for the word in kbl dict, if keyerror return nan'''
            val1 = word_resource_score(word, lexicons) 
            '''word_scores = []
            word_scores.append(val1)'''
            tweet_scores.append(val1) #val1 is a float
        if len(words)==0: 
            val = np.zeros(1)
        else:
            #sum over all words of this tweet
            val = np.nansum(np.array(tweet_scores), axis=0) #here val is a float
        all_scores.append(val)
    return np.array(all_scores) #[num of instances,]


def word_resource_score(word, resource, is_grafs=False, is_embed=False):
    try:
        val = resource[word]
    except KeyError:
        if is_grafs:
            val = [np.nan for i in range(9)]
        elif is_embed:
            val = [np.nan for i in range(300)]
        else:
            val = np.nan
    return val
    
            
def extract_ngrams(tweet, n):
    if n==1:
        return tweet.split()
    return [str(bi[0])+' '+str(bi[1]) for bi in zip(tweet.split()[:-1], tweet.split()[1:])]
