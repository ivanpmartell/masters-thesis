#python create_ct_datasets.py
#python check_spillover.py

#mkdir tts_results
#python train/cnnprom.py --input models/cnnprom_TATA/train.csv --output tts_results/cnnprom_TATA/
python test/cnnprom.py --input models/cnnprom_TATA/test.csv --output tts_results/cnnprom_TATA/results_cnnprom_TATA.csv --model tts_results/cnnprom_TATA/model.pt
python analysis.py --output tts_results/cnnprom_TATA/cnnprom_TATA_results/ --results tts_results/cnnprom_TATA/results_cnnprom_TATA.csv