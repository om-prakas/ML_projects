# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 16:30:15 2019

@author: OmPrakash
"""
# convert python obj into character stream. Serialise obj before writting to file
import pickle  
import numpy as np

with open("train_qa.txt", "rb") as fb:   # Unpickling
    train_data =  pickle.load(fb)
    
with open("test_qa.txt", "rb") as fb:   # Unpickling
    test_data =  pickle.load(fb)
    
' '.join(train_data[0][0])

#create the all word vocabulary
vocabulary = set()  #to make unique element take set
all_data = test_data + train_data
for story, question , answer in all_data:
    vocabulary = vocabulary.union(set(story))
    vocabulary = vocabulary.union(set(question))
vocabulary.add('no')
vocabulary.add('yes')
vocabulary_len = len(vocabulary) + 1 #keras pad sequenece we will use

#find max length of story and question
story_lenths = [len(data[0]) for data in all_data]
max_story_len = max(story_lenths)
max_question_len = max([len(data[1]) for data in all_data])


from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocabulary)

tokenizer.word_index   #All word will be converted to lowercase assiciate with a index

train_story_text = []
train_question_text = []
train_answers = []

for story,question,answer in train_data:
    train_story_text.append(story)
    train_question_text.append(question)

train_story_seq = tokenizer.texts_to_sequences(train_story_text)

#Vectorised your story and questions
def vectorize_stories(data, word_index=tokenizer.word_index, max_story_len=max_story_len,max_question_len=max_question_len): 
    STORY_Vect = []
    QUESTION_Vect = []
    OUTPUT_Vect = []
   
    for story, query, answer in data:      
        # Grab the word index for every word in story
        x = [word_index[word.lower()] for word in story]
        # Grab the word index for every word in query
        y = [word_index[word.lower()] for word in query]
        # Index 0 is reserved so we're going to use + 1 (only yes/no)
        z = np.zeros(len(word_index) + 1)
        z[word_index[answer]] = 1
        
        STORY_Vect.append(x)
        QUESTION_Vect.append(y)
        OUTPUT_Vect.append(z)     
    # RETURN TUPLE FOR UNPACKING
    return (pad_sequences(STORY_Vect, maxlen=max_story_len),pad_sequences(QUESTION_Vect, maxlen=max_question_len), np.array(OUTPUT_Vect))

inputs_train, queries_train, answers_train = vectorize_stories(train_data)
inputs_test, queries_test, answers_test = vectorize_stories(test_data)


tokenizer.word_index['yes']
tokenizer.word_index['no']
sum(answers_test)

#model creation
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout
from keras.layers import add, dot, concatenate
from keras.layers import LSTM

#tupple (max_story_len,batch_size) here batch size not sure so only , 
input_sequence = Input((max_story_len,))
question = Input((max_question_len,))

input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocabulary_len,output_dim=64))
input_encoder_m.add(Dropout(0.3))
#output: (samples, story_maxlen, embedding_dim)

input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocabulary_len,output_dim=max_question_len))
input_encoder_c.add(Dropout(0.3))
#output: (samples, story_maxlen, question_maxlen)

question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocabulary_len,output_dim=64,input_length=max_question_len))
question_encoder.add(Dropout(0.3))
#output: (samples, question_maxlen, embedding_dim)

#pass the input_sequence and question into encoder
input_encoded_m = input_encoder_m(input_sequence) 
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

#question embeded into internal state u #pi = softmax(u^T.mi) 
match = dot([input_encoded_m, question_encoded], axes=(2, 2))
match = Activation('softmax')(match)
# (samples, story_maxlen, query_maxlen)

#add match to 2nd input vector O = Σpi.ci
response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
response = Permute((2, 1))(response)  # (samples, question_maxlen, story_maxlen)

#a = softmax(w(O+U)) #1st concatinate,  weight matrix() then apply softmax
answer = concatenate([response, question_encoded])
answer = LSTM(32)(answer)  # (samples, 32) reduce dimensional
answer = Dropout(0.5)(answer)  
answer = Dense(vocabulary_len)(answer)  # (samples, vocabulary_len)
answer = Activation('softmax')(answer)

#create model
model = Model([input_sequence, question], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()


model_train = model.fit([inputs_train,queries_train],answers_train,
                        batch_size=32,epochs =100,
                validation_data=([inputs_test,queries_test],answers_test))


#plot the accuracy
import matplotlib.pyplot as plt
print(model_train.history.keys())
# summarize history for accuracy
plt.plot(model_train.history['acc'])
plt.plot(model_train.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Input a fact and a question. Predict the answer
pred_results = model.predict(([inputs_test, queries_test]))

test_data 
#rab the 1st test data
test_data[0][0]   # fact
test_data[0][1]   # question
test_data[0][2]   # answer

pred_results.shape    #(1000,38)
#In pred result each word having a probabiility.Find the max probability 

max_value = np.argmax(pred_results[0])  #for 1 result we are checking

for key,value in tokenizer.word_index.items():
    if value == max_value:
        m = key  #print yes/No

pred_results[0][max_value]  #calcualte the % of prediction

#--------------------------
#write your own story using existing vocabulary
vocabulary
my_story = 'John left the kitchen . Sandra dropped the football in the garden .'
my_story.split()
my_question = 'Is the football in office ?'
my_question.split()

#data should be same format as training set
my_data = [(my_story.split(),(my_question.split()),'no')]

my_story_vector,my_que_vector,my_ans_vector = vectorize_stories(my_data)

my_pred_results = model.predict(([ my_story_vector, my_que_vector]))

max_value = np.argmax(my_pred_results[0])  #for 1 result we are checking

for key,value in tokenizer.word_index.items():
    if value == max_value:
        m = key  #print yes/No

print("Predicted answer is: ", m)
print("Probability of certainty was: ", pred_results[0][max_value])

#to save your model
model.save('chatbot_Om.h5')
#load your privious model
#model.load_weights('chatbot_Om.h5')

