import os
import shutil
import json
import re
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys
from nltk.corpus import stopwords
import math
from sentence_transformers import SentenceTransformer, util #, CrossEncoder
from pathlib import Path
from typing import Iterable, Union
import time
def getFolderPath():
    #opens a file explorer window and prompts the user to select location of messages
    app = QApplication(sys.argv) 
    default_dir = os.getcwd()
    path = QFileDialog.getExistingDirectory(None, "Select a directory", default_dir)
    print("Selected path:", path)
    return path

def removeConvos(path, MinSize=16):
    #remove convos is a very dangerous function because 
    # if the path directory is wrong and there are no json files, it will remove everything in it.
    """remove conversation files and folders that are group conversations or just aren't long enough"""
    #path contains to user messages, folders with users names containing their message data.
    msg_1_regex=re.compile("message_1.json$")
    userDir=os.listdir(path)

    #loop through the subfolders in path 
    for dirs in userDir: 
        sub_path=os.path.join(path,dirs)
        if not os.path.isdir(sub_path):
            continue
        # in subfolders check if they contain .json file and open the file.
        # if the number of users is greater than 2, or if theres only one .json and the file size is to small, then remove the .json
        # remove any directories in the subfolders and if there doesn't exist a .json file in it remove the entire subfolder
        hasJson=False
        for f in os.listdir(sub_path):
            f_path=os.path.join(sub_path,f)
            if ".json" in f:
                hasJson=True
                with open(f_path, 'r') as file:
                    data = json.load(file)
                numUsers = len(data["participants"])
                f_size=os.path.getsize(f_path)/1024
                if (f_size < MinSize and re.search(msg_1_regex,f) ) or numUsers > 2:
                    os.remove(f_path)
            elif os.path.isdir(f_path):
                shutil.rmtree(f_path)
            
        if not hasJson:
            print(dirs)
            shutil.rmtree(sub_path)

# Compile a regular expression that matches filenames like:
# "message_1.json", "message_42.json", etc.
# The (\d+) captures the numeric message index.
MSG_RE = re.compile(r"^message_(\d+)\.json$")

def find_message_files(root: Path):
    """
    Recursively find all files named message_<n>.json under the given root
    directory and return them sorted by:
      1) parent directory path
      2) message number n
    """

    found = []  # Will store tuples of (parent_dir, message_index, full_path)

    # Recursively search for any file starting with "message_" and ending in ".json"
    for p in root.rglob("message_*.json"):
        # Attempt to match the filename against the strict regex
        m = MSG_RE.match(p.name)

        if m:
            # Extract the numeric message index from the filename
            msg_index = int(m.group(1))

            # Store parent directory, index, and full path for later sorting
            found.append((p.parent, msg_index, p))

    # Sort first by parent directory, then by message index
    found.sort(key=lambda t: (t[0], t[1]))

    # Return only the Path objects, discarding parent and index
    return [p for _, __, p in found]


def create_output_path(input: Path):  
  """take in file name path of .json messages and 
      create a new path of the output (flattened jsonl messages) in the same location"""
  n = int(MSG_RE.match(input.name).group(1)) #get the number in the file name 
  return input.with_name(f"message_{n}_flat.jsonl") 

notUserMsg=[r" missed your call\.$", r" missed a call from ", r"the video call ended\.$", r"^You called ", r" called you\.$",
            r" removed a message\.$", r" unsent a message\.$", 
            r"Reacted (\\u\w{4})+ to your message ", r" set the nickname for ([A-Z]\w+\s?){1,2} to \'[\D\w]+\'\.$",
            r" sent a location\.$",r" sent a photo\.$", r" sent an attachment\.$", r" sent a GIF\.$", r"^Reacted\s+.+?\s+to your message$"]

def formatAndFilter(data):
    """ function used to extract messages from raw .json data and reformat structure so it is more compatible for training on olama.
    input :
    [{
      "sender_name": "Alex Gedemen",
      "timestamp_ms": 1383042500854,
      "content": "I think",
      "is_geoblocked_for_viewer": false,
      "is_unsent_image_by_messenger_kid_parent": false
    },
    {
      "sender_name": "Alex Gedemen",
      "timestamp_ms": 1383042499247,
      "content": "like will keep their slip",
      "is_geoblocked_for_viewer": false,
      "is_unsent_image_by_messenger_kid_parent": false
    },
    {
      "sender_name": "Saxon Berry",
      "timestamp_ms": 1383042494457,
      "content": "kk ",
      "is_geoblocked_for_viewer": false,
      "is_unsent_image_by_messenger_kid_parent": false
    }]

    output: 
    [{
      "from": "bot",
      "value": "kk"
      "time": 1383042494457, 
      "time2": 0
      },
      "from": "user",
      "value": "like will keep their slip\nI think"
      "time": 1383042500854, 
      "time2": 1607
    }
    """
    messages=data["messages"]
    #NOTE after reversal the last message from 2nd .json file directly preceeds the first message from the 1st .json file
    messages.reverse() #reverse so oldest message is first element and most recent message is last
    ReformatData = []
    #initilise the state of the previous message. {who sent it, the message itself (sequential messages by the same person are combined into the same entry), time stamp of the latest message by user, length of time between the first message in the entry and the last}
    priorMsg = {"from": "", "value": "", "time": 0, "time2": 0}
    notUserMsg_regex=re.compile("|".join(notUserMsg)) # turn notUserMsg list into one string seperated by |
  #compiled_regex_objects = [re.compile(p) for p in notUserMsg]

    for msg in messages:
        # Skip messages without text content
        
        time = msg["timestamp_ms"]
        if "content" not in msg or time < 1325391984000: #ignore content if they aren't a written message or the were sent before 2012 
            continue
                
        value = msg["content"]

        if re.search(notUserMsg_regex, value): #if message is an action e.g. sent attachemnt
            continue
        
        sender = "bot" if msg["sender_name"] == "Saxon Berry" else "user" #rename sender names from their real names to 'user' or to 'bot' if sent by me
        #NotUserMsgBool=[True if intersect(x,value) else False for x in notUserMsg] #find a real function for intersect
        
        #organise messages so if it is sent by user 
        # If same sender and within 60 seconds append message to current value. 
        if priorMsg["from"] == sender and (priorMsg["time"] - time < 60 * 1000):
            ReformatData[-1]["value"] += "\n"+value #add message to previous entry as a new line.
            ReformatData[-1]["time"] = time  #Update time to latest
            ReformatData[-1]["time2"]+= time - priorMsg["time"] #add time elapsed between consecutive messages
            priorMsg = ReformatData[-1].copy()   
        else:
            #if the sender changes or the time between messages is long enough create a new entry.
            newMsg = {"from": sender, "value": value, "time": time, "time2":0} 
            ReformatData.append(newMsg)
            priorMsg = newMsg.copy()
    
    return ReformatData



def simplify4Clustering(msg):
    """simplify sentances to most basic components needed in a sentance.
     remove all non-alphanumeric and non-whitespace characters from the msg string. also remove all stop words e.g. the, and, not
    """  
    msg = re.sub(r'[^\w\s]','',msg)
    msg = msg.encode("ascii", "ignore").decode("ascii")
    msg=msg.lower()
    stop_words = set(stopwords.words("english"))
    #TODO apply lemmatization, which simplifies words to their lemma (google it) 
    return " ".join([word for word in msg.split() if word not in stop_words]) #stop words are things like the, and, not, a, but
    


def splitIn2Convos(messages, time_decay=22, min_time_gap = 5):
    
    #TODO look into more effective ways to split up conversations. Look at "text segmentation with timestamps" gpt chat for recources.
    #NOTE the current method is to calculate the max similarity of the  current message to the previous messages in the active conversation then scale it by time elapsed between these messages. If the resulting value isn't big enough then start new conversation.   

    """Groups message sequences into distinct conversations.
    time_decay: the decay rate (in hours) of the scaling funtion
    min_time_gap: The minimum amount of time elapsed (in minutes) for the current message to be considered part of a new convo"""
    model = SentenceTransformer("all-mpnet-base-v2") #all-MiniLM-L6-v2
    #loops through each message and compares to all the previous messages in the conversation. When requirments fail a message is added to a new list for the next conversation. 
    #requirement: if a message includes a key phrase and long enough time between the next message.

    convos=[]

    for msg in messages:
       
       #msg4nlp contains only the msg information that will be used to train the model or for later preprocessing steps
        msg4nlp = {k: v for k, v in msg.items() if k not in {"time", "time2"}} # new message dict without information about the time they were sent.
        
        embeding = model.encode(msg["value"], convert_to_tensor=True, normalize_embeddings=True) 
        
        #for the first iteration initilise the first entry to the first convo
        if convos == []:
            convos.append([msg4nlp])
            prev_embedings=[embeding]
            currentConvo=[msg]
            continue
        
        #calculates the max similarity of current message with previous messages in the active convo. scales that value by the length of time elapsed between those two messages.  
        max_sim_tScaled =-1 

        for prev_msg, prev_embed in zip(currentConvo, prev_embedings):
            #prev_msg = convos[-1][-1]
            sim = float(util.cos_sim(embeding, prev_embed))
            dt = msg["time"] - prev_msg["time"] #+ prev_msg["time2"]
            similarity_tScaled = sim*math.exp(-1*dt/(time_decay*60*60*1000))
            if similarity_tScaled > max_sim_tScaled:
                #dt_max_sim = dt
                max_sim_tScaled = similarity_tScaled
        dt_lastMsg = msg["time"]-currentConvo[-1]["time"]
        #set the time decay such that max similarity will be less than 0.6 after 15 hours.
        if dt_lastMsg > min_time_gap*60*1000 and max_sim_tScaled < 0.25: #if the reply is after a minimum of min_time_gap minuites and the similarity score is low enough then start new convo.
            convos.append([msg4nlp])
            prev_embedings=[embeding]
            currentConvo=[msg]
        else: #otherwise add the latest message to the existing convo
            convos[-1].append(msg4nlp)
            prev_embedings.append(embeding)
            currentConvo.append(msg)
    return convos
  


def createChunks(convos, window_size = 20):
    """
    window_size of 20 ~ 700 tokens

    Build training examples from chat threads.
    Each example contains up to window_size messages
    Number of training examples for each conversation thread equal to the number of bot replies in it.
    """
    training_dataset = []

    for thread in convos:
        senders = [msg.get("from") for msg in thread]

        # Skip threads that don't contain BOTH user and bot messages.
        participants = set(senders)
        if "bot" not in participants or "user" not in participants:
            continue

        first_user_idx = senders.index("user")

        # Last bot index in the ORIGINAL list (not reversed).
        last_bot_idx = len(senders) - 1 - senders[::-1].index("bot")

        # skip threads where there is no bot reply after the first user message.
        if last_bot_idx <= first_user_idx:
            continue
        
        # Make training window that ends on each bot reply and advances from one bot message to the next bot message.
        # Early replies: stationary window grows from start of thread.
        # Later replies: window becomes fixed-length and slides when it's size equals window_size.
       
        for i, who in enumerate(senders):
            # Only store chunks ending at a bot messages after the first user message.
            if who != "bot" or i <= first_user_idx:
                continue

            start = max(0, i + 1 - window_size) # grows from 0 until it hits window_size, then slides
            training_dataset.append({"conversation": thread[start : i + 1]})

    return training_dataset

def write_jsonl_atomic(lines: Iterable[Union[dict, str]], out_path: Path):
    """Write JSONL atomically; accepts dicts or pre-serialized strings."""
    # Create a temporary path in the same directory so the final replace is atomic
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")

    # Write all lines to the temporary file
    with tmp.open("w", encoding="utf-8") as f:
        for line in lines:
            # If the line is already serialized, write it directly
            # (normalising to exactly one trailing newline)
            if isinstance(line, str):
                f.write(line.rstrip("\n") + "\n")
            else:
                # Otherwise, serialize the dict to JSON
                # ensure_ascii=False preserves Unicode characters
                f.write(json.dumps(line, ensure_ascii=False) + "\n") #convert line from dict to JSON-formatted string and write to tmp file
    
    # override the target JSONL file with the completed tmp file
    os.replace(tmp, out_path)

def process(file_path, overide = True):

    out = create_output_path(file_path)
    if out.exists() and not overide:
        print("exists: ", out)
        return
    
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    msg_data = formatAndFilter(data) #preprocess fb messenger json data. reformat content for traning or later preprocessing steps and and remove irrelevent data not needed for those. 

    for msg in msg_data: msg["value2"] = simplify4Clustering(msg["value"]) #create a simplified version of each reply to use for clustering in later preprocessing steps

    convos = splitIn2Convos(msg_data) #group message threads into distinct conversations

    traningdata = createChunks(convos, window_size = 15) #within each conversation thread create training samples. 
    
    write_jsonl_atomic(traningdata, out) #save training samples as jsonl file. 

    

if __name__=="__main__":
    
    path = Path(getFolderPath())

    jsonList = find_message_files(path)
    
    for json_path in jsonList:
        UserName = json_path.parent.name
        filename= json_path.stem
        #try:
        start_time = time.perf_counter()
        process(json_path)
        timeElapsed = time.perf_counter() - start_time
        print(f'Completed {filename} of {UserName}')
        print(f"time to process {timeElapsed}")
        #except Exception as e:
            #print(f"ERROR processing {json_path}: {e}")

