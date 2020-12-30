import numpy as np
from nltk.tokenize import word_tokenize
from collections import defaultdict
import os

# Used to map a string label to an int
def mapStrToInt( listString ):

    '''Returns a dict to map each unique string to an int'''

    mappings = { label:num+1 for (num,label) in enumerate( sorted( set(listString) ) ) }
    return mappings


# Used to get counts of unique words in each file
def countWords( listFiles ):

    '''
    Returns:
        A list of dicts containing the counts of each unique word per file
        A dict containing the total counts of each unique word across all files
    '''

    # 'lambda:0' ensures that if a key is not in the dict, default value is 0
    # Could also just use 'int', but this is more explicit
    totalCounts = defaultdict( lambda:0 )   # Word counts across all files
    fileCounts = []                         # List of word counts per file

    # Read each training file
    for fileTrain in listFiles:

        fileCounts.append( defaultdict( lambda:0 ) )
        with open( os.path.join( os.getcwd(), fileTrain ), 'r' ) as f:

            fContents = f.read()

            for word in word_tokenize( fContents ):
            
                # If word has not been previously seen, count is 0
                totalCounts[word] += 1
                fileCounts[-1][word] += 1

    return fileCounts, totalCounts


def calcSimilarity( inputs, centroids ):

    '''
    Arguments:
        inputs -> A matrix, where each row represents the weighted word counts for a single document
        centroids -> A matrix, where each row represents the averaged weighted word counts for a single class
    
    Returns:
        The cosine similarities between each row of inputs and the classes
    '''

    # Add a dimension for broadcasting
    dotProd = np.sum( inputs * centroids, axis=-1 )[:, np.newaxis]
    norm = ( np.linalg.norm( inputs, axis=-1 ) * np.linalg.norm( centroids, axis=-1 ) )[:, np.newaxis]

    similarity =  dotProd / norm

    return similarity 


if __name__ == "__main__":

    print( "Please input all filepaths relative to the current working directory:\n", os.getcwd(), sep='', end='\n\n' )

    if __debug__:
        # fileTrainList = "corpus1_train.labels"
        fileTrainList = "../train.txt"
    else:
        fileTrainList = input( "Please input the name of the file containing the list of labeled training files.\n" )
    
    listTrainFile = []
    listTrainLabel = []

    # Read file to get list of training files & labels
    with open( os.path.join( os.getcwd(), fileTrainList ), 'r' ) as f:
        
        for line in f:

            path, label = line.split()
            listTrainFile.append( path )
            listTrainLabel.append( label )

    # Number of documents
    N = len( listTrainFile )

    # Get dict for mapping and convert labels from str -> int
    mappingsLabels = mapStrToInt( listTrainLabel )
    reverseMappings = { v:k for (k,v) in mappingsLabels.items() }
    listTrainLabel = list( map( mappingsLabels.get, listTrainLabel ) )
    
    dictLabelToFile = defaultdict( list )
    for i in range( len( listTrainLabel ) ):
        dictLabelToFile[ listTrainLabel[i] ].append(i) 

    # Get word counts for each training file
    fileCounts, totalCounts = countWords( listTrainFile )

    # Get mappings for words from str -> int
    mappingsWords = mapStrToInt( totalCounts.keys() )

    # Map words 
    totalCounts = { mappingsWords[k]:v for (k,v) in totalCounts.items() }
    for i in range( len(fileCounts) ):
        fileCounts[i] = { mappingsWords[k]:v for (k,v) in fileCounts[i].items() }
    numTotalWords = len( totalCounts.keys() )

    # Calculate idf for each word
    idf = np.zeros( (numTotalWords,1) )
    for i in range( N ):
        idf[ np.array(list( fileCounts[i].keys() ))-1 ] += 1
    idf = np.log10( N / idf )

    # Calculate weighted tf-idf value
    weighted = []
    for i in range( N ):
        weighted.append( { k:( np.log10(v+1) * idf[k-1] ).item() for (k,v) in fileCounts[i].items() } )

    # Convert dict to array
    arrWeighted = np.zeros( (N,numTotalWords) )
    for i in range( N ):
        for key in weighted[i].keys():
            arrWeighted[i][key-1] = weighted[i][key]

    # Calculate centroid of each class
    centroids = np.zeros( (len( dictLabelToFile.keys() ), numTotalWords) )
    for key in dictLabelToFile.keys():

        listDocs = dictLabelToFile[key]
        centroids[key-1] = np.average( arrWeighted[ listDocs ], axis=0 )

    if __debug__:
        # fileTestList = "corpus1_test.list"
        fileTestList = "../test.txt"
    else:
        fileTestList = input( "Please input the name of the file containing the list of testing files.\n" )
    
    listTestFile = []
    # Read file to get list of test files
    with open( os.path.join( os.getcwd(), fileTestList ), 'r' ) as f:
        for line in f:
            path, label = line.split()
            listTestFile.append( path )

    # Get word counts for each testing file
    testCounts, _ = countWords( listTestFile )

    # Ignores a word if has not been previously seen
    # B/c unknown words map to an idf value of 0
    weightedTest = []
    for i in range( len(testCounts) ):
        testCounts[i] = { mappingsWords[k]:v for (k,v) in testCounts[i].items() if k in mappingsWords }
        weightedTest.append( { k:( np.log10(v+1) * idf[k-1] ).item() for (k,v) in testCounts[i].items() } )

    arrWeightedTest = np.zeros( ( len(weightedTest), numTotalWords ) )
    for i in range( len(testCounts) ):
        for key in weightedTest[i].keys():
            arrWeightedTest[i][key-1] = weightedTest[i][key]

    # Add a dimension for broadcasting
    arrWeightedTest = arrWeightedTest[:, np.newaxis, :]

    # Calculate cosine similarity
    similarity = calcSimilarity( arrWeightedTest, centroids )

    # Maximize the similarity and extract predictions from nested list
    # E.g. [ [x],[y],[z] ] -> [ x,y,z ]
    pred = np.argmax( similarity, axis=-1 )
    pred = [ ind[0] for ind in pred ]
    names = [ reverseMappings[i+1] for i in pred ]

    # Save predictions to a file
    with open( os.path.join( os.getcwd(), "../", "preds.txt" ), 'w' ) as f:
        for i in range( len( listTestFile ) ):
            f.write( listTestFile[i] + " " + names[i] + "\n" )