import pandas as pd
import numpy as np
import re
from keras.preprocessing import text, sequence


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
d1=int(df.shape[0]*70/100)
d2=int(df.shape[0]*90/100)
train= df.iloc[0:d1,:]
train.to_csv('trainsp', sep='\t', encoding='utf-8')
validation= df.iloc[d1:d2,:]
train.to_csv('trainval', sep='\t', encoding='utf-8')
test=df.iloc[d2:,:]
train.to_csv('testval', sep='\t', encoding='utf-8')

#test = pd.read_csv('test.csv')

print (train)

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
    
replacements=appos = {
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
    


 
x_train = train["comment_text"].fillna("fillna").values
for i  in range(0,d1):
    x_train[i] = clean(x_train[i])
print ("OK x")

x_test = test["comment_text"].fillna("fillna").values
for i  in range(0,d2-d1):
    x_train[i] = clean(x_train[i])
print ("OK y")

embeddings_index = {}
with open('glove.txt',encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs 
        
        

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_train = train[list_classes].values
print (x_train)


max_features=20000
tokenizer = text.Tokenizer(num_words=max_features, filters='"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(list(x_train) + list(x_test))
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
 
# pad sentences to meet the maximum length
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test= sequence.pad_sequences(x_test, maxlen=maxlen)

# build the embedding matrix 

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector


def build_model(): # figure (a) as a Keras model
    inp = Input(shape=(maxlen, ))
    
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = GlobalMaxPool1D()(x)
    print (x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    outp = Dense(6, activation="sigmoid")(x)
    
    model = Model(inputs=inp, outputs=outp)
 
    return model

model = build_model()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=1000,
          epochs=1,
          validation_split=0.1)


##serialize model to JSON
#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
## serialize weights to HDF5
#model.save_weights("model.h5")
#print("Saved model to disk")
#
#
# load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")
# 
## evaluate loaded model on test data
#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))