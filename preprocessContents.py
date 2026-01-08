import os
import shutil
import json
import re
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys
from nltk.corpus import stopwords
import math
from sentence_transformers import SentenceTransformer, util

def getFolderPath():
    #opens a file explorer window and prompts the user to select location of messages
    app = QApplication(sys.argv) 
    default_dir = os.getcwd()
    path = QFileDialog.getExistingDirectory(None, "Select a directory", default_dir)
    print("Selected path:", path)
    return path

def removeConvos(path, MinSize=16):
    """remove conversation files and folders that are group conversations or just aren't long enough"""
    #path contains to user messages, folders with users names containing their message data.
    msg_1_regex=re.compile("message_1.json$")
    
    #loop through the subfolders in path 
    for dirs in os.listdir(path): 
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
  n=int(MSG_RE.match(input.name).group(1)) #get the number in the file name 
  return input.with_name(f"message_{n}_flat.jsonl") 

notUserMsg=[" missed your call\.$", " missed a call from ", "the video call ended\.$", "^You called ", " called you\.$",
            " removed a message\.$", " unsent a message\.$", 
            "Reacted (\\u\w{4})+ to your message ", " set the nickname for ([A-Z]\w+\s?){1,2} to \'[\D\w]+\'\.$",
            " sent a location\.$"," sent a photo\.$", " sent an attachment\.$", " sent a GIF\.$"]

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
    messages=messages.reverse() #reverse so oldest message is first element and most recent message is last
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
        #If same sender and within 120 seconds append message to current value. 
        if priorMsg["from"] == sender and (priorMsg["time"] - time < 120 * 1000):
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
    " ".join([word for word in msg.split() if word not in stop_words]) #stop words are things like the, and, not, a, but
    #TODO apply lemmatization, which simplifies words to their lemma (google it)


def splitIn2Convos(messages, time_decay=30):
    #TODO look into more effective ways to split up conversations. Look at "text segmentation with timestamps" gpt chat for recources.
    #NOTE the current method is to calculate the max similarity of the most recent message to the messages in the active conversation then scale it by time elapsed between these messages. If the resulting value isn't big enough then start new conversation.   
    #NOTE this is problematic because it only considers one message in the convo. If the entire convo has similar messages but the oldest one is most similar the time scaling might push it below the threshold.
    """Groups message sequences into distinct conversations"""
    model = SentenceTransformer("all-MiniLM-L6-v2")
      
    #loops through each message and compares to all the previous messages in the conversation. When requirments fail a message is added to a new list for the next conversation. 
    #requirement: if a message includes a key phrase and long enough time between the next message.

    convos=[]

    for msg in messages:
       
       #msg4nlp contains only the msg information that will be used to train the model or for later preprocessing steps
        msg4nlp = {k: v for k, v in msg.items() if k not in {"time", "time2"}} # new message dict without information about the time they were sent.
        
        embeding = model.encode(msg["value"], convert_to_tensor=True, normalize_embeddings=True) 
        
        #for the first iteration initilise the first entry to the first convo
        if convos is []:
            convos.append([msg4nlp])
            prev_embedings=[embeding]
            continue
        
        #calculates the max similarity of current message with previous messages in the active convo. scales that value by the length of time elapsed between those two messages.  
        max_similarity=-1 
        for prev_msg, prev_embed in zip(convos[-1], prev_embedings):
            #prev_msg = convos[-1][-1]
            simB4 = float(util.cos_sim(embeding, prev_embed))

            if simB4 > max_similarity:
                max_similarity = simB4
                dt = msg["time"] - prev_msg["time"] + prev_msg["time2"]
      
        similarity_tScaled=max_similarity*math.exp(-1*dt/time_decay*60*1000)
        #set the time decay such that max similarity will be less than 0.6 after 15 hours.

        if dt > 5*60*1000 and similarity_tScaled < 0.6: #if the reply is after 5 mins and the similarity score is low enough then start new convo.
            convos.append([msg4nlp])
            prev_embedings=[]
        else: #otherwise add the latest message to the existing convo
            convos[-1].append(msg4nlp)
            prev_embedings.append(embeding)
    return convos
  
def createChunks(convos, wSize, wStep):
    traningdataset=[]
    for thread in convos:
      sender=[x["from"] for x in thread]
      #
      
      if not("bot" in sender and "user" in sender):  #skip conversations with only one user 
            continue
      elif sender[::-1].index("bot")<sender.index("from"): #index of the last message from bot is less than the first from user. aka the bot never replies to the user. 
            continue
      
      chunk=[]

      botMsgIdx=[i for i, x in enumerate(sender) if x=="bot"] #create list index of all the bots messages in the convo.

      for i in botMsgIdx:
        if i>0 and i <= wSize:
          chunk.append({"conversation":thread[:i+1]})
        elif i > wSize:
          chunk.append({"conversation":thread[i + 1 - wSize : i + 1]})
      
      if chunk:
        traningdataset.extend(chunk)

    return traningdataset

def write_jsonl_atomic(lines: Iterable[Union[dict, str]], out_path: Path):
    """Write JSONL atomically; accepts dicts or pre-serialized strings."""
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for line in lines:
            if isinstance(line, str):
                f.write(line.rstrip("\n") + "\n")
            else:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
    tmp.replace(out_path)

def process(file_path):

    out = create_output_path(file_path)
    if out.exists:
        print("exists: ", out)
        return
    
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    msg_data = formatAndFilter(data)

    for msg in msg_data: msg["value2"] = simplify4Clustering(msg["value"])

    convos = splitIn2Convos(msg_data)

    traningdata = createChunks(convos)
    
    write_jsonl_atomic(traningdata, out)

    

if __name__=="__main__":
    
    path = getFolderPath()

    removeConvos(path)

    jsonList = find_message_files(path)

    for json_path in jsonList:
        try:
            process(json_path)
        except Exception as e:
            print(f"ERROR processing {json_path}: {e}")

