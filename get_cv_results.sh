mkdir cv_results
python cross_validate/cnnprom.py --input datasets/cnnprom_TATA/dataframe.csv --output cv_results/cnnprom_TATA/results_cnnprom_TATA.csv
python cross_validate/cnnprom.py --input datasets/cnnprom_nonTATA/dataframe.csv --output cv_results/cnnprom_nonTATA/results_cnnprom_nonTATA.csv
python cross_validate/cnnprom.py --input datasets/cnnprom_complete/dataframe.csv --output cv_results/cnnprom_complete/results_cnnprom_complete.csv

python cross_validate/icnn.py --input datasets/icnn_complete/dataframe.csv --output cv_results/icnn_complete/results_icnn_complete.csv

python cross_validate/dprom.py --input datasets/dprom_TATA/dataframe.csv --output cv_results/dprom_TATA/results_dprom_TATA.csv
python cross_validate/dprom.py --input datasets/dprom_nonTATA/dataframe.csv --output cv_results/dprom_nonTATA/results_dprom_nonTATA.csv
python cross_validate/dprom.py --input datasets/dprom_complete/dataframe.csv --output cv_results/dprom_complete/results_dprom_complete.csv