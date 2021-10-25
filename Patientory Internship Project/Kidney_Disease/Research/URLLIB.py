import http.client
import json
import urllib.request
import urllib.parse


url_prefix = 'https://patientory.com/'
headers = {
                'Client_Id': 'b122dfd830ff2698acdf6b4dfb942a27',
                'Client_Secret': 'd0f595fbad966d26c8564e0380966126',    
             }

req = urllib.request.Request(url_prefix,headers = headers )
json_set = json.loads(urllib.request.urlopen(req).read().decode('utf-8')) 
