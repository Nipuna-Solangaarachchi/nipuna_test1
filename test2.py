import pandas as pd
import numpy as np
from keras.preprocessing import text, sequence


from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input 
from keras.layers import concatenate
from keras.layers import GlobalAveragePooling1D,Embedding,GlobalMaxPooling1D,SpatialDropout1D,Bidirectional,TimeDistributed,Dropout,LSTM
from keras.models import Model
from keras.initializers import Constant

#from keras.preprocessing.text import Tokenizer


VALIDATION_SPLIT = 0.2
 
maxlen = 200 # length of the submitted sequence
#EMBEDDING_FILE = './data/fasttext/crawl-300d-2M.vec'

embed_size=100
