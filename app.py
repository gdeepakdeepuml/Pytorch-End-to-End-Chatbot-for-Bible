import streamlit as st
import torch 
import json 
import random 
import pandas as pd 
import streamlit.components.v1 as stc 
import codecs
from model import NeuralNet
from nltk_util import tokenization,stemm,bag_of_word

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open('bibleversa.json','r') as f:
    intents = json.load(f)

File = 'bible2.pth'
data = torch.load(File)

input_size = data['input_size']
output_size = data['output_size']
hidden_size = data['hidden_size']
model_state = data['model_state']
all_words = data['all_words']
tags = data['tags']

model = NeuralNet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()


#print("do you help to protect from covid!! let's chat! type 'quit' to get out !! ")

def chatpost(msg):
#while True:
    #sentence = input("You: ")
    #print(msg)
    sentence = msg
    if sentence.lower() == "quit":
        print("its work's") 

    sentence = tokenization(sentence)
    x = bag_of_word(sentence,all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)

    output = model(x)
    _, predicted = torch.max(output,dim=1)
    tag = tags[predicted.item()]
    #print(tag)
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    #print(prob.item())
    responses = None
    if prob.item() > 0.65:
        for intent in intents['intents']:
            if tag == intent['tag']:
                #print(tag)
                responses = intent['responses']
        #print(responses)      
        return (random.choice(responses))
    else:
        answer="I don't understand can you be more clear please!"
        return (answer)

#msh = "i am so Depressed"
#result = chatpost(msh)

#print(result)

    
def get_text():
    input_text = st.text_input("You:")
    return input_text 

#st.markdown(html_temp,unsafe_allow_html=True)

html_bible = """
<div style=height:250px;width=50%;text-align:center;background:#1abc9c;padding: 60px;color: white;>
  <h1 style=font-size:80px;color: white;>Bible</h1>
  <p style=font-size:25px;>Is not my word like a fire 🔥</p>
</div>

<div padding:20px;>
  <h1>All about me</h1>
  <p style=font-size:20px>Hi i am Bible ,i am with a compilation of 66 books and letters written by more than 40 authors during a period of approximately 1,500 years, my original text was communicated in just three languages: Hebrew, koine or common Greek, and Aramaic and The Old Testament was written for the most part in Hebrew, with a small percentage in Aramaic. The New Testament was written in Greek.</p>
  <p style=font-size:20px>Do you know meaning of your name 🙃 , but i know The English word "Bible" comes from bíblia in Latin and bíblos in Greek. The term means book, or books, and may have originated from the ancient Egyptian port of Byblos (in modern-day Lebanon), where papyrus used for making books and scrolls was exported to Greece. Other terms for the Bible are the Holy Scriptures, Holy Writ, Scripture, or the Scriptures, which means "sacred writings."</p>
  <p style=font-size:20px>i have two main Sections -- the Old and New Testament -- I am with several more divisions: the Pentateuch, the Historical Books, the Poetry and Wisdom Books, the books of Prophecy, the Gospels, and the Epistles.</p> 
  <p style=font-size:20px>this is just version one in next version i will show you the word of god, i mead the book Bible 😊</p>
</div>

"""
@st.cache(allow_output_mutation=True)
def user_input_list():
    return []

st.markdown(html_bible,unsafe_allow_html=True)

st.write("this is version of this BIBLE-BOT which you can use this for topics like Forgiveness,Depression,Failure,salvation,hope and faith,prayers from Bible 😊")

st.subheader("You can start your conversation with Bible")
user_input = get_text()
result = chatpost(user_input)

#user_input_list.append(user_input)
#bible_ans.append(chatpost(user_input))
if st.button("send"):
    st.text_area("Bible:",value=result,
                 height=400,max_chars=None,key=None)
    user_input_list().append({"user_input":user_input,"bible_ans":result})
else:
    st.text_area("Bible:",value="before we start lets pray \n\n Father God, our heart is filled with chaos and confusion. We feel as if we am drowning in our circumstances and our heart is filled with fear and confusion. We really need the strength and peace that only You can give. Right now, we  choose to rest in You. In Jesus’ Name I pray, Amen. Now read ",
                 height=300,max_chars=None,key=None)
    
#st.subheader("YOU: \t"+ user_input)
#st.subheader("Bible: \n"+chatpost(user_input))
#print(user_input)
df = pd.DataFrame(user_input_list()) 

st.subheader("You conversation with Bible 😊")

for i in range(len(df)):
    if len(df) == 0:
        st.write("start you converzation")
    else:
        val = df.values[i]
        st.subheader("YoU: \t"+str(val[1]))
        st.subheader("Bible: \n"+str(val[0]))