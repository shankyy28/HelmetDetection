import requests
from datetime import datetime

url = 'http://your-flask-app-url/processImage'
files = {'image': open(r"E:\Code\CV\images\bike_1.jpg", 'rb')}
data = {
    'timestamp': datetime.now().isoformat(),
    'address': '123 Main St, Anytown, USA'
}

response = requests.post(url, files=files, data=data)
print(response.json())