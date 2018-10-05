# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 13:43:33 2018

@author: Nipuna
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

def Confidence_Intent(q):
    confidence_intent=q["intent"]["confidence"]
    return confidence_intent

def Trigger_Intent(q): 

    tr_intent=q["intent"]["name"]

    tr_entities=[i["entity"] for i in q['entities']]

    tr_crf_entity_dic={}
    tr_duckling_entity_list=[]
    entities = q["entities"]
    for entity in entities:
        if (entity["extractor"] == "ner_duckling") & (entity["entity"] == "time"):
            tr_duckling_entity_list.append([entity["additional_info"]["grain"],entity["start"],entity["end"],entity["value"]])
        if entity["extractor"] == "ner_crf":
            tr_crf_entity_dic[entity["entity"]]=[entity["value"],entity["start"],entity["end"]]
    return tr_intent,tr_entities,tr_duckling_entity_list,tr_crf_entity_dic



def JSONLoad(urlHead,tr_intent):  
    tr_intent=tr_intent.replace(":","_")
    print (tr_intent)
    url=urlHead+tr_intent+".json"
    with open(url,encoding='utf-8') as data_file:
        json_intent = json.loads(data_file.read())
    return json_intent

def getOutput(json_intent,tr_intent,tr_entities,tr_duckling_entity_list,tr_crf_entity_dic,confidence_intent):
    parameters=json_intent["responses"][0]["parameters"]
    entities_list=[i["name"] for i in parameters]
    print ("entities_list ",entities_list)
    print ("tr_entities ",tr_entities)
    print ("tr_crf_entity_dic ",tr_crf_entity_dic)
    print ("tr_duckling_entity_list ",tr_duckling_entity_list)
    
    #tr_entities_req=(list(set(tr_entities) & set(entities_list)))
    index_arr=[]
    for i in range(len(entities_list)):
        if entities_list[i] in tr_crf_entity_dic.keys():
            index_arr.append(i)
       
    print (index_arr)  
    #para_val_list=[json_intent["responses"][0]["parameters"][i]["value"] for i in index_arr]
    
    lst=[]
    for i in index_arr:
        dict={}
        value=json_intent["responses"][0]["parameters"][i]["value"]
        name=json_intent["responses"][0]["parameters"][i]["name"]
        if ".original" in value:
            for j in tr_duckling_entity_list:
                if ((tr_crf_entity_dic[name][1]==j[1]) & (tr_crf_entity_dic[name][2]==j[2])):
                    return_value=j[3]
                else:
                    return_value=""
            
        else:
          return_value=tr_crf_entity_dic[name][0]
        dict["entity"]=name
        dict["value"]=return_value
        lst.append(dict)

    intent_action=json_intent["responses"][0]["action"]
    response={"intent":tr_intent,"parameters":lst,"action":intent_action,"confidence":confidence_intent}
    return response


def train_nlu(data, configure, model_dir):
    training_data = load_data(data)
    trainer = Trainer(config.load(configure))
    trainer.train(training_data)
    model_directory = trainer.persist(model_dir,fixed_model_name="weatherNLU3")
    
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
        
        confidence_intent= Confidence_Intent(a)
        if confidence_intent >= 0.7:
                tr_intent,tr_entities,tr_duckling_entity_list,tr_crf_entity_dic= Trigger_Intent(a)
                json_intent=JSONLoad("DialogBot/intents/",tr_intent)
                response=getOutput(json_intent,tr_intent,tr_entities,tr_duckling_entity_list,tr_crf_entity_dic,confidence_intent)   

        else:
            response = {
                        "intent": "default.intent",
                        "parameters":[],
                        "message":" "
                        }
        
        print(response)

 
