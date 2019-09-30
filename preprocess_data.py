import os
import pandas as pd

from pydub import AudioSegment


# Dev set
DEV_INPUT_PATH = "D:/Do An/Datasets/Breath_datasets_wav/Training/developement/data/"
DEV_OUTPUT_PATH = "D:/Do An/Datasets/Breath_datasets_wav/Training/developement/"
DEV_LABEL_PATH = "D:/Do An/Datasets/Breath_datasets_wav/Training/developement/label/"

# Test set
TEST_INPUT_PATH = "D:/Do An/Datasets/Breath_datasets_wav/Training/validation/data/"
TEST_OUTPUT_PATH = "D:/Do An/Datasets/Breath_datasets_wav/Training/validation/"
TEST_LABEL_PATH = "D:/Do An/Datasets/Breath_datasets_wav/Training/validation/label/"



CHUNK = 5
OVERLAP = 2
OUTPUT_FOLDER = 'output'
LABEL_FOLDER = 'label'
 # Type of breath
BREATH_TYPE = ['normal', 'deep', 'strong']

def check_directory(origin_path, folder):
    directory = os.path.join(origin_path, folder)
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_audio_segment (filename, source_path, destination_path, start, end, status, chunk, overlap):
    """[summary]
    
    Arguments:
        filename {[type]} -- [filename of the audio file]
        source_path {[type]} -- [source of a audio file]
        destination_path {[type]} -- [destination of a output file]
        start {[type]} -- [description]
        end {[type]} -- [description]
        status {[type]} -- [type of breath]
        chunk {[type]} -- [length of the segment]
        overlap {[type]} -- [overlap %]
    """
    
    # Pydub works in milliseconds
    start = start * 1000 
    end = end * 1000 
    
    # Get the audio file
    src_Audio = AudioSegment.from_wav(source_path)
    
    #Cut the right part
    output_audio = src_Audio[start:end]
    
    #split it into 5s each and then export 
    lenInSec = ( len(output_audio)/(chunk*1000)).__round__() #split each path by chunk second
    
    start = 0
    metaInfo=[["",""]]
    for i in range(0, lenInSec):
        audio_partition = output_audio[start * 1000: (start + chunk) * 1000]
        fname =  filename + str(i) + "-" + status + '.wav'
        audio_partition.export(destination_path + '' + fname, 
                         format="wav")  # Exports to a wav file in the current path.
        start += chunk/overlap
        # start += chunk


def split_by_label(source_path, destination_path, label_path):

    # Check the output directory status 

    check_directory(destination_path, OUTPUT_FOLDER)

    output_path = os.path.join(destination_path, OUTPUT_FOLDER) + '/'
    # Get all the audio files
    filenames = os.listdir(source_path)
    
    meta_data=[["",""]]
    
    # Go through all the file 
    for filename in filenames:
        
        # take the file name without dot
        filename =  filename.split(".")[0]
        
        #get wav file name path
        wav_path = source_path + filename + ".wav"
        
        #get label filename path
        csv_path = label_path + filename + ".txt"
        
        #read the label file path 
        label = pd.read_csv(csv_path, delim_whitespace= True)
        
        #Normal breath
        for breath in BREATH_TYPE:
            if (breath in label['Status'].values):
                breath_start = label[label['Status'] == breath]['Start'].iloc[0]
                breath_end   = label[label['Status'] == breath]['End'].iloc[-1]               
                
                #Export the file
                get_audio_segment(filename, wav_path, output_path, breath_start, breath_end, breath, CHUNK, OVERLAP)


# Devset
split_by_label(DEV_INPUT_PATH, DEV_OUTPUT_PATH, DEV_LABEL_PATH)

# Testset
split_by_label(TEST_INPUT_PATH, TEST_OUTPUT_PATH, TEST_LABEL_PATH)




