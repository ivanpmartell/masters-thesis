python cross_validate/cnnprom.py
python cross_validate/icnn.py
python cross_validate/dprom.py

python train/cnnprom.py --output trained/cnnprom --input models/cnnprom/train.csv
python train/icnn.py --output trained/icnn --input models/icnn/train.csv
python train/dprom.py --output trained/dprom --input models/dprom/train.csv

