import requests
import json
import time
import os
import sys




class DataMiner:
    
    name = ""
    
    api_key = 'Kq9KsoDOYdLZ0vWEGxiDVRL0021FLBI2'
    baseUrl = f'https://api.nytimes.com/svc/search/v2/articlesearch.json?q=health&api-key={api_key}'
    
   
    
    
    
    
    def __init__(self, filename):
        
        self.name = filename+'.json'
    
    
    def requestData(self):
        
        try:
            data = requests.get(self.baseUrl, timeout=5).json()
        except(requests.ConnectionError,requests.Timeout) as exception:
            
            temp = os.getcwd()
            
            if self.name in os.listdir(temp):
                
                os.remove(self.name)
                
                sys.exit()
            
            
        
        
        time.sleep(10)
        
        outFile = open(self.name,'w')
        
        json.dump(data,outFile,indent=6)
        
        outFile.close()
    
    def extractArticle(self):
        
        f = open(self.name,'r')
        
        data = json.load(f)
        
       
        
        lst = data['response']['docs']
        
        f.close()
        
        return lst
    
    def imageList(self):
        
        temp = self.extractArticle()
        img = []
        
        for i in range(0,len(temp)):
            
            if i == 4:
                continue;
            else:
                
                img.append('https://www.nytimes.com/'+temp[i]['multimedia'][2]['url'])
           
    
            
        return img
    
    
    
        
        
    
        
        
        
        
        
        
    
    









