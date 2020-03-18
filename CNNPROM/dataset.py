import pathlib
import numpy as np
from Bio import SeqIO
import pandas as pd
import random
from skorch.dataset import Dataset
import mysql.connector as mariadb
from tqdm import tqdm

class CNNPROMDataset(Dataset):
    seqs_length: int
    y_type: np.dtype = np.int64
    dataframe: pd.DataFrame
    dna_dict: dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    lbl_dict: dict = {'Non-Promoter': 0, 'Promoter': 1}

    def __init__(self, file, neg_file, num_negatives, binary, save_df):
        if('csv' in pathlib.Path(file).suffix):
            self.load_dataframe(file)
        else:
            seqs = self.load_file(file)
            self.seqs_length = len(seqs[0])
            df = pd.DataFrame(seqs, columns=['sequence'])
            df['label'] = self.lbl_dict['Promoter']

            if(neg_file is not None):
                neg_seqs = self.load_file(neg_file)
                neg_seqs_length = len(neg_seqs[0])
                if(self.seqs_length != neg_seqs_length):
                    raise Exception(r"Promoter and Non-Promoter sequence lengths don't match")
            else:
                print("Preprocessing: Creating the negative sequences")
                neg_seqs = self.create_negative_seqs(num_negatives)
            neg_df = pd.DataFrame(neg_seqs, columns=['sequence'])
            neg_df['label'] = self.lbl_dict['Non-Promoter']
            self.dataframe = df.append(neg_df, ignore_index=True)
        if(binary):
            self.y_type = np.float32
        if(save_df):
            self.save_dataframe('models/cnnprom/dataframe.csv')

    def load_file(self, file):
        records = []
        with open(file, 'rU') as fasta_file:
            for record in SeqIO.parse(fasta_file, 'fasta'):
                r_seq = record.seq._data
                if 'N' not in r_seq:
                    records.append(r_seq)
        return records

    def load_dataframe(self, file):
        self.dataframe = pd.read_csv(file)
        example = self.dataframe.iloc[0]
        self.seqs_length = len(example.sequence)
    
    def save_dataframe(self, file):
        print('Saving dataframe to: %s' % file)
        self.dataframe.to_csv(file, index=False)
    
    def create_negative_seqs(self, num_negatives):
        mariadb_connection = mariadb.connect(host='genome-mysql.soe.ucsc.edu', user='genomep', password='password', database='hg38')
        cursor = mariadb_connection.cursor()
        neg_seqs = []
        cursor.execute(r"SELECT chrom, strand, txStart, txEnd, exonStarts, exonEnds, name2 FROM `refGene` AS rg WHERE (rg.name LIKE 'NM_%' AND rg.chrom NOT LIKE '%\_%' AND rg.exonCount > 1) ORDER BY RAND() LIMIT " + str(num_negatives))
        table = cursor.fetchall()
        mariadb_df = pd.DataFrame(table, columns=cursor.column_names)
        cursor.close()
        mariadb_connection.close()
        while(len(neg_seqs) < num_negatives):
            for c in tqdm(mariadb_df.chrom.unique()):
                chrom_file = 'data/human_chrs/%s.fa' % c
                mariadb_chrom_df = mariadb_df.loc[mariadb_df['chrom'] == c]
                with open(chrom_file, 'rU')as cf:
                    chrom_seq = SeqIO.read(cf, 'fasta')
                    for _, row in mariadb_chrom_df.iterrows():
                        try:
                            if row['strand'] == '-':
                                exon1_end = int(row['exonEnds'].split(',')[-2]) - 1
                                gene_end = row['txStart'] + self.seqs_length
                                seq = 'N'
                                while 'N' in seq:
                                    random_neg = random.randint(gene_end, exon1_end)
                                    seq = chrom_seq.seq[random_neg - self.seqs_length: random_neg]
                                    seq = seq.upper()
                                neg_seq = self.create_antisense_strand(seq)
                            else:
                                exon1_end = int(row['exonEnds'].split(',')[0]) + 1
                                gene_end = row['txEnd'] - self.seqs_length
                                seq = 'N'
                                while 'N' in seq:
                                    random_neg = random.randint(exon1_end, gene_end)
                                    seq = chrom_seq.seq[random_neg: random_neg + self.seqs_length]
                                    seq = seq._data.upper()
                                neg_seq = seq
                            neg_seqs.append(neg_seq)
                            if(len(neg_seqs) >= num_negatives):
                                break
                        except Exception as e:
                            print('Error processing %s: %s' % (row['name2'], str(e)))
                if(len(neg_seqs) >= num_negatives):
                    break
            mariadb_df = mariadb_df.sample(frac=1).reset_index(drop=True)
        return neg_seqs

    reversal = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    def create_antisense_strand(self, org_seq):
        negs = []
        for letter in org_seq[::-1]:
            negs.append(self.reversal[letter])
        result = ''.join(negs)
        return result

    def one_hot_encoder(self, seq):
        one_hot = np.zeros((len(self.dna_dict), len(seq)), dtype=np.float32)
        for idx, token in enumerate(seq):
            one_hot[self.dna_dict[token], idx] = 1
        return one_hot
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        x = self.one_hot_encoder(row.sequence.upper())
        return x, np.array(row.label, dtype=self.y_type)
    
    def __len__(self):
        return len(self.dataframe)