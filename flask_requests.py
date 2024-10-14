import requests

url = "http://127.0.0.1:5000/create-embeddings"
data = {"image_folder": "../Belajar-DeepFace"}

response = requests.post(url, json=data)

print(response.json())  # Print the response

url = "http://127.0.0.1:5000/initialize-faiss"
response = requests.post(url)
print(response.json())

url = "http://127.0.0.1:5000/recognize-face"
response = requests.get(url)
print(response.json())
