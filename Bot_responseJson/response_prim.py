# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 13:38:14 2018

@author:Nipuna
"""

from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer
from rasa_nlu import config
from rasa_nlu.model import Interpreter

import random
import json

null="null"
true="true"
false="false"


def Trigger_Intent(q): 
    confidence_intent=q["intent"]["confidence"]
    tr_intent=q["intent"]["name"]
        
    return tr_intent,confidence_intent
    

def JSONLoad(urlHead,tr_intent):    
    url=urlHead+tr_intent+".json"
    with open(url,encoding='utf-8') as data_file:
        json_intent = json.loads(data_file.read())
    return json_intent


def getOutput(json_intent):
    parameters=(json_intent["responses"][0]["parameters"])
    message_list=(json_intent["responses"][0]["messages"][0]["speech"])
    
    len_msg_lst=len(message_list)
    response={}
    response["intent"]=json_intent["name"]
    response["parameters"]=parameters
    
    if len_msg_lst==0:
      response["message"]=""
    else:
      rand_int=random.randint(0,len_msg_lst-1)
      rand_message=message_list[rand_int]
      response["message"]=rand_message

    return (response)
  
  
def train_nlu(data, configure, model_dir):
    training_data = load_data(data)
    trainer = Trainer(config.load(configure))
    trainer.train(training_data)
    model_directory = trainer.persist(model_dir,fixed_model_name="mod_name")
    
def get_input():
    userInput = input("Comment : ")
    return userInput
    
def run_nlu():
    userIn = get_input()    
    a = interpreter.parse(userIn)
    return a
  
if __name__ == '__main__':
    #train_nlu('./data/data.json', 'config.yml', './models/nlu')
    interpreter = Interpreter.load('./Pro/models/default/Mod1/')
    while(True):
        a = run_nlu()
        print(a)
        
        tr_intent,confidence_intent= Trigger_Intent(a)
        if confidence_intent >= 0.1:
          json_intent=JSONLoad("./Testbot_prim/intents/",tr_intent)
          response=getOutput(json_intent)   

        else:
            response = {
                        "intent": "default.intent",
                        "parameters":[],
                        "message":" "
                        }
        
        print(response)
