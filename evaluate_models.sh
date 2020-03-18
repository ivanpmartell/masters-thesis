mkdir evaluation
mkdir evaluation/cnnprom
# CNNPROM trained model on DPROM dataset
python test/cnnprom.py --output evaluation/cnnprom/dprom.csv --input models/dprom/dataframe.csv
python analysis.py --output evaluation/cnnprom/dprom_results/ --results evaluation/cnnprom/dprom.csv
# CNNPROM trained model on ICNN dataset
python test/cnnprom.py --output evaluation/cnnprom/icnn.csv --input models/icnn/dataframe.csv
python analysis.py --output evaluation/cnnprom/icnn_results/ --results evaluation/cnnprom/icnn.csv

mkdir evaluation/dprom
# DPROM trained model on CNNPROM dataset
python test/dprom.py --output evaluation/dprom/cnnprom.csv --input models/cnnprom/dataframe.csv
python analysis.py --output evaluation/dprom/cnnprom_results/ --results evaluation/dprom/cnnprom.csv
# DPROM trained model on ICNN dataset
python test/dprom.py --output evaluation/dprom/icnn.csv --input models/icnn/dataframe.csv
python analysis.py --output evaluation/dprom/icnn_results/ --results evaluation/dprom/icnn.csv

mkdir evaluation/icnn
# ICNN trained model on CNNPROM dataset
python test/icnn.py --output evaluation/icnn/cnnprom.csv --input models/cnnprom/dataframe.csv
python analysis.py --output evaluation/icnn/cnnprom_results/ --results evaluation/icnn/cnnprom.csv
# ICNN trained model on DPROM dataset
python test/icnn.py --output evaluation/icnn/dprom.csv --input models/dprom/dataframe.csv
python analysis.py --output evaluation/icnn/dprom_results/ --results evaluation/icnn/dprom.csv