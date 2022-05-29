import requests
import bs4
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
r = requests.get('https://www.imaware.health/blog/most-common-gastrointestinal-conditions')
bs = BeautifulSoup(r.text,'html.parser')

for i in bs.select('p'):
    print(i.text)
