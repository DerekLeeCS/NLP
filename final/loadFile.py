import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import os
from typing import List, Dict

# Load the list of files
def loadList( fileList: str ) -> tuple( [ List[str], List[str] ] ):

    listFiles = []
    listLabels = []

    # Read file to get list of files & labels
    with open( os.path.join( os.getcwd(), fileList ), 'r' ) as f:
        
        for line in f:

            path, label = line.split()
            listFiles.append( path )
            listLabels.append( label )

    return listFiles, listLabels


# Used to map a string label to an int
def mapStrToInt( listString: List[str] ) -> Dict[str,int]:

    '''Returns a dict to map each unique string to an int'''

    mappings = { label:num+1 for (num,label) in enumerate( sorted( set(listString) ) ) }
    return mappings


# Create mappings given a list of labels
def createMappings( listLabels: List[str] ) -> tuple( [ Dict[str,int], Dict[int,str] ] ):

    # Get dict for mapping 
    mappingsLabels = mapStrToInt( listLabels )
    reverseMappings = { v:k for (k,v) in mappingsLabels.items() }

    return mappingsLabels, reverseMappings


def labeler(example, index):
    return example, tf.cast(index, tf.int64)


# Used to get counts of unique words in each file
def tokenizeFile( listFiles: List[str], mappedLabels: List[int] ):
    
    # Load the dataset with each line as an example
    dsLabeled = []
    for fileName, label in zip( listFiles, mappedLabels ):

        text = tf.data.TextLineDataset( fileName )
        ex = text.map( lambda line: labeler(line, label))
        dsLabeled.append( ex )

    dsText = dsLabeled[0]
    for labeled_dataset in dsLabeled[1:]:
        dsText = dsText.concatenate(labeled_dataset)

    BUFFER_SIZE = 50000
    dsText = dsText.shuffle(
        BUFFER_SIZE, reshuffle_each_iteration=False)

    # Map the vocabulary
    VOCAB_SIZE=10000
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=VOCAB_SIZE)
    encoder.adapt( dsText.map(lambda text, label: text) )

    # Encode the dataset
    # encoded_example = [ encoder( line ) for line, _ in dsText ]
    # print( encoded_example )

    return dsText, encoder#, encoded_example