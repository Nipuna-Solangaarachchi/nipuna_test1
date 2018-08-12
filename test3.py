# Comment is added
import pandas as pd
import numpy as np
from keras.preprocessing import text, sequence
import re

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input 
from keras.layers import concatenate
from keras.layers import GlobalAveragePooling1D,Embedding,GlobalMaxPooling1D,SpatialDropout1D,GlobalMaxPool1D,Bidirectional,TimeDistributed,Dropout,LSTM
from keras.models import Model
from keras.initializers import Constant

#from keras.preprocessing.text import Tokenizer





VALIDATION_SPLIT = 0.2
 
maxlen = 200 # length of the submitted sequence
#EMBEDDING_FILE = './data/fasttext/crawl-300d-2M.vec'

embed_size=100

df = pd.read_csv('train.csv')
df = df.sample(frac=1).reset_index(drop=True)
d1=int(df.shape[0]*10/100)
d2=int(df.shape[0]*90/100)
train= df.iloc[0:d1,:]
train.to_csv('trainsp', sep='\t', encoding='utf-8')
validation= df.iloc[d1:d2,:]
train.to_csv('trainval', sep='\t', encoding='utf-8')
test=df.iloc[d2:,:]
train.to_csv('testval', sep='\t', encoding='utf-8')

#test = pd.read_csv('test.csv')



print (train)

 
x_trainz = train["comment_text"].fillna("fillna").values

print (x_trainz[0])

def clean(comment):
    """
    This function receives comments and returns a clean comment
    """
    comment=comment.lower()
    # normalize ip
    comment = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"," ip ", comment)
    # replace words like Gooood with Good
    comment = re.sub(r'(\w)\1{2,}', r'\1\1', comment)
    # replace ! and ? with <space>! and <space>? so they can be kept as tokens by Keras
    comment = re.sub(r'(!|\?)', " \\1 ", comment)   
 
    #Split the sentences into words
    words=comment.split(' ')
 
    # normalize common abbreviations
    # replacements is a dictionary loaded from https://drive.google.com/file/d/0B1yuv8YaUVlZZ1RzMFJmc1ZsQmM/view 
    words=[replacements[word] if word in replacements else word for word in words]
 
    clean_sent=" ".join(words)
    return(clean_sent)
    
replacements={
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not"
}
    
print (clean(x_trainz[0]))
