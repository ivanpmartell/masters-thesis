import os
import pathlib
import numpy as np
from Bio import SeqIO
import pandas as pd
from skorch.dataset import Dataset

class ICNNDataset(Dataset):
    seqs_start = 249
    elements_length: int
    non_elements_length: int
    y_type: np.dtype = np.int64
    dataframe: pd.DataFrame
    dna_dict: dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    lbl_dict: dict = {'Non-Promoter': 0, 'Promoter': 1}

    def __init__(self, file, neg_folder, num_positives, binary, save_df):
        if('csv' in pathlib.Path(file).suffix):
            self.load_dataframe(file)
        else:
            seqs = self.load_file(file)
            seqs_length = len(seqs[0])
            df = pd.DataFrame(seqs, columns=['sequence'])
            df['label'] = self.lbl_dict['Promoter']
            sample_df = df.sample(n=num_positives)

            self.dataframe = sample_df
            for neg_file in self.get_files_in_folder(neg_folder):
                neg_seqs = self.load_file(neg_file)
                neg_seqs_length = len(neg_seqs[0])
                if(seqs_length != neg_seqs_length):
                    raise Exception(r"Promoter and Non-Promoter sequence lengths don't match")
                neg_df = pd.DataFrame(neg_seqs, columns=['sequence'])
                neg_df['label'] = self.lbl_dict['Non-Promoter']
                self.dataframe = self.dataframe.append(neg_df, ignore_index=True)
        
        example = self.dataframe.iloc[0]
        element, non_element = self.sequence_encoder(example.sequence.upper())
        self.elements_length = len(element)
        self.non_elements_length = non_element.shape[1]
        if(binary):
            self.y_type = np.float32
        if(save_df):
            self.save_dataframe('models/icnn/dataframe.csv')

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
    
    def save_dataframe(self, file):
        print('Saving dataframe to: %s' % file)
        self.dataframe.to_csv(file, index=False)

    @staticmethod
    def get_files_in_folder(folder):
        file_list = []
        for (dirpath, _, filenames) in os.walk(folder):
            file_list.extend([os.path.join(dirpath, file) for file in filenames if not "readme" in file])
        print(file_list)
        return file_list

    def one_hot_encoder(self, seq):
        one_hot = np.zeros((len(self.dna_dict), len(seq)), dtype=np.float32)
        for idx, token in enumerate(seq):
            one_hot[self.dna_dict[token], idx] = 1
        return one_hot
    
    def element_encoder(self, seq):
        element = np.zeros((len(seq), len(self.dna_dict)), dtype=np.float32)
        for idx, token in enumerate(seq):
            element[idx, self.dna_dict[token]] = 1
        return element.flatten()

    def GC_CAAT_Box_extractor(self, seq):
        return seq[self.seqs_start -110:self.seqs_start -70 + 1] #139:180
    
    def TFIIB_TATA_Box_extractor(self, seq):
        return seq[self.seqs_start -37:self.seqs_start -25 + 1] #212:225

    def initiator_extractor(self, seq):
        return seq[self.seqs_start -2:self.seqs_start +4 + 1] #247:254

    def DCEI_extractor(self, seq):
        return seq[self.seqs_start +6:self.seqs_start +11 + 1] #255:261
    
    def DCEII_DCEIII_DPE_extractor(self, seq):
        return seq[self.seqs_start +28:self.seqs_start +34 + 1] #277:284

    def element_extractor(self, seq):
        return self.GC_CAAT_Box_extractor(seq) + self.TFIIB_TATA_Box_extractor(seq) +\
               self.initiator_extractor(seq) + self.DCEI_extractor(seq) + self.DCEII_DCEIII_DPE_extractor(seq)

    def non_element_extractor(self, seq):
        return seq[:self.seqs_start -110] + seq[self.seqs_start -70 + 1:self.seqs_start -37] +\
               seq[self.seqs_start -25 + 1:self.seqs_start -2] + seq[self.seqs_start +4 + 1:self.seqs_start +6] +\
               seq[self.seqs_start +11 + 1:self.seqs_start +28] + seq[self.seqs_start +34 + 1:]
    
    def svm_encoder(self, seq):
        enc = np.zeros(len(seq), dtype=np.float32)
        for idx, token in enumerate(seq):
            enc[idx] = self.dna_dict[token]
        return enc

    def sequence_encoder(self, seq):
        non_element = self.non_element_extractor(seq)
        element = self.element_extractor(seq)
        encoded_non_element = self.one_hot_encoder(non_element)
        encoded_element = self.element_encoder(element)
        return encoded_element, encoded_non_element

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        encoded = self.sequence_encoder(row.sequence.upper())
        return encoded, np.array(row.label, dtype=self.y_type)
    
    def __len__(self):
        return len(self.dataframe)