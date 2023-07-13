- As it is a small project, I kept multiple classes in a single file for simplicity, but on a larger 
project, ideally they would have their own file. The same goes for training and prediction, 
ideally they should have their own module.
- Hardcoded information like file names and the API key were kept on the repository only for testing purposes.
### How to run
After cloning the repository, open the property-friends folder
on a terminal and run the following commands (Docker must be installed on your machine):
```
docker build -t property-friends .
```
```
docker run -p 8000:8000 property-friends
```
Add the train.csv and test.csv files to the root folder of the repository. 
On a second terminal instance, run the following command to train the model.
```
curl -v -X 'POST' \
'http://0.0.0.0:8000/train' \
-H 'accept: application/json' \
-H 'Content-Type: application/json' \
-H "X-API-Key: a8de2bbd-c3cb-4b0e-bc49-c872c8eb40e6" \
-d '{
"train_path": "train.csv",
"test_path": "test.csv"
}'
```
After that, predictions can be made with the following command:
```
curl -X 'POST' \
  'http://0.0.0.0:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: a8de2bbd-c3cb-4b0e-bc49-c872c8eb40e6" \
  -d '{
  "type": "casa",
  "sector": "vitacura",
  "net_usable_area": 152.0,
  "net_area": 257.0,
  "n_rooms": 3,
  "n_bathroom": 3,
  "latitude": -33.3794,
  "longitude": -70.5447
}'
```