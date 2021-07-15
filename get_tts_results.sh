#python create_ct_datasets.py
#python check_spillover.py

#mkdir tts_results
## CNNPROM RESULTS
#python train/cnnprom.py --input models/cnnprom_TATA/train.csv --output tts_results/cnnprom_TATA/
#python test/cnnprom.py --input models/cnnprom_TATA/test.csv --output tts_results/cnnprom_TATA/results_cnnprom_TATA.csv --model tts_results/cnnprom_TATA/model.pt
#python analysis.py --output tts_results/cnnprom_TATA/cnnprom_TATA_results/ --results tts_results/cnnprom_TATA/results_cnnprom_TATA.csv
#python test/cnnprom.py --input models/dprom_TATA/test.csv --output tts_results/cnnprom_TATA/results_dprom_TATA.csv --model tts_results/cnnprom_TATA/model.pt
#python analysis.py --output tts_results/cnnprom_TATA/dprom_TATA_results/ --results tts_results/cnnprom_TATA/results_dprom_TATA.csv

python train/cnnprom.py --input models/cnnprom_nonTATA/train.csv --output tts_results/cnnprom_nonTATA/
python test/cnnprom.py --input models/cnnprom_nonTATA/test.csv --output tts_results/cnnprom_nonTATA/results_cnnprom_nonTATA.csv --model tts_results/cnnprom_nonTATA/model.pt
python analysis.py --output tts_results/cnnprom_nonTATA/cnnprom_nonTATA_results/ --results tts_results/cnnprom_nonTATA/results_cnnprom_nonTATA.csv
python test/cnnprom.py --input models/dprom_nonTATA/test.csv --output tts_results/cnnprom_nonTATA/results_dprom_nonTATA.csv --model tts_results/cnnprom_nonTATA/model.pt
python analysis.py --output tts_results/cnnprom_nonTATA/dprom_nonTATA_results/ --results tts_results/cnnprom_nonTATA/results_dprom_nonTATA.csv

python train/cnnprom.py --input models/cnnprom_complete/train.csv --output tts_results/cnnprom_complete/
python test/cnnprom.py --input models/cnnprom_complete/test.csv --output tts_results/cnnprom_complete/results_cnnprom_complete.csv --model tts_results/cnnprom_complete/model.pt
python analysis.py --output tts_results/cnnprom_complete/cnnprom_complete_results/ --results tts_results/cnnprom_complete/results_cnnprom_complete.csv
python test/cnnprom.py --input models/dprom_complete/test.csv --output tts_results/cnnprom_complete/results_dprom_complete.csv --model tts_results/cnnprom_complete/model.pt
python analysis.py --output tts_results/cnnprom_complete/dprom_complete_results/ --results tts_results/cnnprom_complete/results_dprom_complete.csv
python test/cnnprom.py --input models/icnn_complete/test.csv --output tts_results/cnnprom_complete/results_icnn_complete.csv --model tts_results/cnnprom_complete/model.pt
python analysis.py --output tts_results/cnnprom_complete/icnn_complete_results/ --results tts_results/cnnprom_complete/results_icnn_complete.csv

#TODO: ICNN and DPROM