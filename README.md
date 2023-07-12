- As it is a small project, I kept all the classes for the
model training in a single file for simplicity, but on a larger 
project, ideally they would have their own file.
- requirements.txt
- The tests can be run with a smaller version of the original 
dataset for performance
- Hardcoded file paths can be changed to cloud storage resource names
- .env and API key were kept on the repository only for testing purposes
```
curl -v -X 'POST' \
'http://127.0.0.1:8000/train' \
-H 'accept: application/json' \
-H 'Content-Type: application/json' \
-H "X-API-Key: a8de2bbd-c3cb-4b0e-bc49-c872c8eb40e6" \
-d '{
"train_path": "model/train.csv",
"test_path": "model/test.csv"
}'
```