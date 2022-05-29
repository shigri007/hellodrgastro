

import tqdm
import requests
import cgi




from tqdm import tqdm





buffer_size = 1024





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







