
import tqdm
import requests
import cgi




from tqdm import tqdm

from flask import Flask,jsonify, render_template, request, flash, redirect,session


from DataPipeline.dataMiner import DataMiner
import mysql.connector 
import os
import json
import cv2

import tensorflow
from tensorflow.keras.models import load_model
import numpy as np


model1 = load_model('model/Predictor.model')






def Mpredict(path):
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb= cv2.resize(image_rgb,(224,224))
    image_rgb= image_rgb.reshape(1,224,224,3)
    
    return np.argmax(model1.predict(image_rgb))





import nltk
#nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

import requests
import bs4
from bs4 import BeautifulSoup
import pandas as pd

from keras.models import load_model    
model = load_model('model.h5')
import json
import random
intents = json.loads(open('data1.json', encoding="utf8").read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))



def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res










app = Flask(__name__)


def predict_label(img_path):
    
	pass

@app.get('/img')
def img():
    return render_template('image.html')



@app.post('/sub')
def predict():
    
    if request.method == 'POST':
        
        img = request.files['image']
        img_path = 'static/'+img.filename
        img.save(img_path)
        
        
        p = Mpredict(img_path)
    return render_template('image.html',prediction=p,img_path = img_path)
        
        




obj = DataMiner('data')

lst = obj.extractArticle()
img = obj.imageList()
app.secret_key=os.urandom(24)

conn=mysql.connector.connect(host="localhost", user="root", password="", database="user")
cursor=conn.cursor()


app.static_folder = 'static'

@app.route("/chat")
def chat():
    return render_template("chatbot.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)
  

@app.route('/signup')
def signup():
     if 'user_id' in session:
      return render_template('dashboard.html')
     else:
      return render_template('signup.html')
   


@app.route('/' )
def home():
    
    if 'user_id' in session:
      
      return render_template('dashboard.html')
    else:
        return redirect('/signup')
@app.route('/admin panel' )
def select():
    if 'user_id1' in session:
        cursor.execute("SELECT * from details")
        data=cursor.fetchall()
        return render_template('admin panel.html', data = data)
    else:
        return render_template('admin.html')



	   
	    

@app.route('/login' )
def login():
    
     if 'user_id' in session:
      return render_template('dashboard.html')
     else:
      
      
      return render_template('login.html')
      



@app.route('/validation' , methods=['Post'])
def val():
   
    email=request.form.get('email')
    password=request.form.get('password')
    cursor.execute( "SELECT * FROM details WHERE email LIKE '{}' AND password LIKE '{}'".format(email,password))
    users=cursor.fetchall()
    if len(users)>0:
       
        session['user_id']=users[0][0] 
        session['username']=users[0][1]
        return redirect('/')
    else :
         flash(u'Invalid Email or Password', 'error')
         return redirect('/login')
@app.route('/ad' )
def adpage():
    return render_template('admin.html')


@app.route('/admin' , methods=['Post'])
def adminval():
   
    email=request.form.get('email')
    password=request.form.get('password')
    cursor.execute( "SELECT * FROM admin WHERE username LIKE '{}' AND password LIKE '{}'".format(email,password))
    users1=cursor.fetchall()
    if len(users1)>0:
        session['user_id1']=users1[0][0] 
        session['username']=users1[0][1]
       
        
        return redirect('/admin panel')
    else :
       
          
         return redirect('/admin')


@app.route('/deleteForm')###Delete Route 
def show_delete():
	return render_template("delete_id.html")
@app.route('/deleteid',methods=['POST','GET'])##Delete Values based on ID
def delete_data():
	
    upid=request.form.get('upid')
    cursor.execute("DELETE FROM details WHERE id=%s", (upid,))
   
    return render_template('link2.html')





@app.route('/add', methods=['POST'])
def add():
    fname=request.form.get('first')
    lname=request.form.get('last')   
    email=request.form.get('Email')
    password=request.form.get('password')
    cursor.execute("INSERT INTO  details (id,firstname,lastname,email,password) VALUES (NULL,'{}','{}','{}','{}')".format(fname,lname,email,password))
    conn.commit()
    cursor.execute( "SELECT * FROM details WHERE email LIKE '{}' ".format(email))
    myuser=cursor.fetchall()
    session['user_id']=myuser[0][0] 
    session['username']=myuser[0][1]
        
    return redirect('/')

@app.route('/logout')
def logout():
    session.pop('user_id')
    return redirect('/login')
@app.route('/logout1')
def logout1():
    session.pop('user_id1')
    return redirect('/ad')




@app.route('/chat', methods=['GET','POST'])
def chatPage():
    return render_template('chat.html')


       
@app.get('/news')
def index():
    if 'user_id' in session:
    
    
    
        return render_template('news.html',artic=lst,im=img)
    else:
        return redirect('/signup')
@app.route('/virtual')
def virtual():
    if 'user_id' in session:
    
    
    
        return render_template('virtual.html',artic=lst,im=img)
    else:
        return redirect('/signup')



@app.get('/disease')
def index1():
    
    return render_template('index.html')



@app.get('/abdo')
def dbPage():
    return render_template('abdominal pain.html')

@app.get('/diarrhea')
def dPage():
    return render_template('Acute diarrhea.html')
@app.get('/fissures')
def fPage():
    return render_template('Anal fissures.html')
@app.get('/tract')
def tPage():
    return render_template('Biliary Tract Disorders, Gallbladder Disorders, and Gallstone Pancreatitis.html')

@app.get('/Cons')
def conPage():
    return render_template('Constipation and Defecation Problems.html')
@app.get('/stones')
def sPage():
    return render_template('Gallstones.html')
@app.get('/gas')
def gPage():
    return render_template('gas.html')
@app.get('/gerd')
def gerdPage():
    return render_template('gerd.html')
@app.get('/indigestion')
def iPage():
    return render_template('indigestion.html')
@app.get('/irritable')
def irrPage():
    return render_template('Irritable Bowel Syndrome.html')
@app.get('/lactose')
def lacPage():
    return render_template('Lactose Intolerance.html')
@app.get('/liver')
def liverPage():
    return render_template('Liver Disease.html')
@app.get('/loss')
def lossPage():
    return render_template('loss of apetite.html')
@app.get('/ulcer')
def ulcerPage():
    return render_template('ulcer.html')
@app.get('/vomit')
def vomitPage():
    return render_template('vomiting.html')











    
    
    
@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/after1')
def ab():
    return render_template('after1.html')
@app.route('/after2')
def ab2():
    return render_template('after2.html')

@app.route('/after3')
def ab3():
    return render_template('after3.html')

@app.route('/after4')
def ab4():
    return render_template('after4.html')

@app.route('/after5')
def ab5():
    return render_template('after5.html')

@app.route('/after6')
def ab6():
    return render_template('after6.html')

@app.route('/after7')
def ab7():
    return render_template('after7.html')

@app.route('/after8')
def ab8():
    return render_template('after8.html')

@app.route('/after9')
def ab9():
    return render_template('after9.html')

@app.route('/after10')
def ab10():
    return render_template('after10.html')


buffer_size = 1024
@app.route('/imgex')
def imgx():

    url = 'https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip'
    response = requests.get(url, stream=True)
    flename = ""
    file_size = int(response.headers.get('Content-Length',0))

    default_filename = url.split('/')[-1]

    disposition = response.headers.get('Content-Disposition')

    if disposition:


    
        value, params = cgi.parse_header(disposition)
    
        filname = params.get('filename',default_filename)
    else:
        filename = default_filename

    progress = tqdm(response.iter_content(buffer_size),f"Downloading : {filename}", total=file_size,unit='B',unit_scale=True,unit_divisor=1024)

    with open(filename,'wb') as f:
            
       for data in progress.iterable:
                   
          f.write(data)
        
          progress.update(len(data))


    return render_template('admin panel.html')
    
  




@app.route('/dataex')
def dataex():
    r = requests.get('https://www.imaware.health/blog/most-common-gastrointestinal-conditions')
    bs = BeautifulSoup(r.text,'html.parser')

    for i in bs.select('p'):

       print(i.text)


    return render_template('admin panel.html')





if __name__ == '__main__':
    app.run(debug=True)
 