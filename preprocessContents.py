import os
import shutil
import json
path = r"C:\Users\saxon\OneDrive\Documents\facebook messages\e2ee_cutover 3"#inbox 3"

def removeSmallconvos(path,MinSize=16):
    #path contains to user messages, folders with users names containing their message data.
    for dirs in os.listdir(path):

        sub_path=os.path.join(path,dirs)
        if not os.path.isdir(sub_path):
            continue
        hasJson=False
        for f in os.listdir(sub_path):
            f_path=os.path.join(sub_path,f)
            if ".json" in f:
                hasJson=True
                with open(f_path, 'r') as file:
                    data = json.load(file)
                numUsers = len(data["participants"])
                f_size=os.path.getsize(f_path)/1024
                if f_size < MinSize or numUsers > 2:
                    os.remove(f_path)
            elif os.path.isdir(f_path):
                shutil.rmtree(f_path)
            
        if not hasJson:
            print(dirs)
            shutil.rmtree(sub_path)

#reformat messages into readable format readable by llm 
""" {
  "conversations": [
    {"from": "user", "value": "hey you around?"},
    {"from": "bot", "value": "yeah what's up?"},
    {"from": "user", "value": "keen for jam today?"},
    {"from": "bot", "value": "always keen if you're there 😉"}
  ]
} """
#current format
"""
  "participants": [
    {
      "name": "James Fraser"
    },
    {
      "name": "Saxon Berry"
    }
  ],
  "messages": [
    {
      "sender_name": "James Fraser",
      "timestamp_ms": 1718368145728,
      "content": "Reacted \u00f0\u009f\u0098\u00a2 to your message ",
      "is_geoblocked_for_viewer": false,
      "is_unsent_image_by_messenger_kid_parent": false
    },
    {
      "sender_name": "Saxon Berry",
      "timestamp_ms": 1718351635800,
      "content": "Bro U can have dinner any night. Lazer force is once in a life time",
      "reactions": [
        {
          "reaction": "\u00f0\u009f\u0098\u00a2",
          "actor": "James Fraser",
          "timestamp": 1718368145
        }
      ],
      "is_geoblocked_for_viewer": false,
      "is_unsent_image_by_messenger_kid_parent": false
    }, """
#combine multiple messages in a row from the same user. 
with open(f_path, 'r') as file:
    data = json.load(file)

messages=data["messages"]
messages=messages.reverse()
trainData = []
priorMsg = {"from": "", "value": "", "time": 0, "time2":0}

#use spaCy to do identify similarity in these key phrases.
notUserMsg=["Reacted \u00f0\u009f\u0098\u00a2 to your message ", "James missed your call.","You missed a call from James.","You called James.", "Saxon removed a message.","Adam sent a location.","Saxon sent a photo.",] 
#remove messages such as video calls, attatchemnt sent e.c.t.

for msg in messages:
    # Skip messages without text content
    
    time = msg["timestamp_ms"]
    if "content" not in msg or time < 1325391984000: #before 2012
        continue    
    sender = "bot" if msg["sender_name"] == "Saxon Berry" else "user"
    value = msg["content"]
    
    NotUserMsgBool=[True if intersect(x,value) else False for x in notUserMsg] #find a real function for intersect
    
    # If same sender and within 10 seconds
    if priorMsg["from"] == sender and (priorMsg["time"] - time < 120 * 1000):
        # Append message to current value
        trainData[-1]["value"] += "\n"+value
        trainData[-1]["time"] = time  # Update time to latest
        trainData[-1]["time2"]+= time - priorMsg["time"]
        priorMsg = trainData[-1].copy()   

    elif True==NotUserMsgBool:
        continue
    else:
        # Create a new message group
        newMsg = {"from": sender, "value": value, "time": time, "time2":0}
        trainData.append(newMsg)
        priorMsg = newMsg.copy()

#find conversations. If they span over a single day they are part of the same conversation.
#conversations need to have at least one response by me to the user.
    # no conversations 2 in length started by me
    # must have both partys in conversation
#All conversation threads must be ended by me.
from sentence_transformers import SentenceTransformer, util
import math
def splitIn2Convo(messages,time_decay=30):
  model = SentenceTransformer("all-MiniLM-L6-v2")
#loops through each message and compares to all the previous messages in the conversation. When requirments fail a message is added to a new list for the next conversation. 
#requirement: if a message includes a key phrase and long enough time between the next message.
#requirment: someone replys with a thumbs up emoji
#scenario: 
  convo=[]

  for msg in messages:
    if convo is []:
      convo.append([msg])
      prev_embeding=model.encode(msg["value"])
      continue
    prev_msg = convo[-1][-1]
    dt = msg["time"] - prev_msg["time"] + prev_msg["time2"]

    embeding=model.encode(msg["value"],convert_to_tensor=True, normalize_embeddings=True)
    #set the time decay such that max similarity will be less than 0.6 after 15 hours.
    similarity= float(util.cos_sim(embeding, prev_embeding))
    similarity_tScaled=similarity*math.exp(-dt/time_decay*60*1000)

    if dt > 5*60*1000 or similarity_tScaled < 0.6: 
        convo.append([msg])
    else:
        convo[-1].append(msg)
  
def createChunks(convos, wSize, wStep):
    for thread in convos:
        if len(thread)<3 and thread[0]["from"]=="bot":
          continue
        sender=[x["from"] for x in thread]
        if not("bot" in sender and "user" in sender) or :
            continue
            

#go through directories and extract .json file
#get name of parent folder
#delete any folder that doesn't have a .json file bigger than 15kb
#delete any that are younger than 2012
#rename the jason file so it is the name of the parent folder in the case of double ups 
