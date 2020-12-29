import numpy as np
from nltk.tokenize import word_tokenize
import os
from collections import defaultdict


def mapStrToInt( listString ):

    mappings = { label:num+1 for (num,label) in enumerate( sorted( set(listString) ) ) }
    return mappings


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

    # Get mappings for labels from str -> int
    mappingsLabels = mapStrToInt( listTrainLabel )
    reverseMappings = { v:k for (k,v) in mappingsLabels.items() }
    print( mappingsLabels )

    # Map labels 
    listTrainLabel = list( map( mappingsLabels.get, listTrainLabel ) )
    
    dictLabelToFile = defaultdict( list )
    for i in range( len( listTrainLabel ) ):
        dictLabelToFile[ listTrainLabel[i] ].append(i) 

    # 'lambda:0' ensures that if word has not been previously seen, default value is 0
    # Could also just use 'int', but this is more explicit
    totalCounts = defaultdict( lambda:0 )   # Word counts for all files
    fileCounts = []                         # List of word counts for each file

    # Read each training file
    for fileTrain in listTrainFile:

        fileCounts.append( defaultdict( lambda:0 ) )
        with open( os.path.join( os.getcwd(), fileTrain ), 'r' ) as f:

            fContents = f.read()

            for word in word_tokenize( fContents ):
            
                # If word has not been previously seen, count is 0
                totalCounts[word] += 1
                fileCounts[-1][word] += 1

    
    # tupleCounts = sorted(totalCounts.items(), key=lambda x: x[1], reverse=True)
    # print( tupleCounts )

    # Get mappings for words from str -> int
    mappingsWords = mapStrToInt( totalCounts.keys() )

    # Map words 
    totalCounts = { mappingsWords[k]:v for (k,v) in totalCounts.items() }
    for i in range( len(fileCounts) ):
        fileCounts[i] = { mappingsWords[k]:v for (k,v) in fileCounts[i].items() }

    # print( mappingsWords )

    # tupleCounts = sorted(totalCounts.items(), key=lambda x: x[1], reverse=True)
    # print( tupleCounts )

    totalWords = len( totalCounts.keys() )

    # Calculate idf for each word
    idf = np.zeros( (totalWords,1) )
    for i in range( N ):
        idf[ np.array(list( fileCounts[i].keys() ))-1 ] += 1
    idf = np.log10( N / idf )

    print( idf )

    # print( idf )

    # Calculate weighted tf-idf value
    weighted = []
    for i in range( N ):
        weighted.append( { k:( np.log10(v+1) * idf[k-1] ).item() for (k,v) in fileCounts[i].items() } )

    # print( weighted )
    # print( len(weighted) )

    # Convert dict to array
    arrWeighted = np.zeros( (N,totalWords) )

    # Loop through each document and the weighted word counts
    for i in range( N ):
        for key in weighted[i].keys():
            arrWeighted[i][key-1] = weighted[i][key]

    # print( dictLabelToFile )

    # Calculate centroid of each class
    centroid = np.zeros( (len( dictLabelToFile.keys() ), totalWords) )
    for key in dictLabelToFile.keys():

        listDocs = dictLabelToFile[key]
        centroid[key-1] = np.average( arrWeighted[ listDocs ], axis=0 )

    # print( centroid )


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

    # Count for each testing file
    testCounts = []                         # List of word counts for each file

    # Read each testing file
    for fileTest in listTestFile:

        testCounts.append( defaultdict( lambda:0 ) )
        with open( os.path.join( os.getcwd(), fileTest ), 'r' ) as f:

            fContents = f.read()
            
            for word in word_tokenize( fContents ):
            
                # If word has not been previously seen, count is 0
                testCounts[-1][word] += 1

    # Ignores a word if has not been previously seen
    # B/c unknown words map to an idf value of 0
    weightedTest = []
    for i in range( len(testCounts) ):
        testCounts[i] = { mappingsWords[k]:v for (k,v) in testCounts[i].items() if k in mappingsWords }
        weightedTest.append( { k:( np.log10(v+1) * idf[k-1] ).item() for (k,v) in testCounts[i].items() } )

    arrWeightedTest = np.zeros( ( len(weightedTest), totalWords ) )
    for i in range( len(testCounts) ):
        for key in weightedTest[i].keys():
            arrWeightedTest[i][key-1] = weightedTest[i][key]

    # print( np.shape( np.linalg.norm( centroid, axis=-1 ) ) )
    # print( np.shape(  np.linalg.norm( arrWeightedTest, axis=-1 )) )
    # print( np.shape( arrWeightedTest * centroid ) )
    arrWeightedTest = arrWeightedTest[:, np.newaxis, :]
    # print( np.shape( np.sum( arrWeightedTest * centroid, axis=-1 )[:, np.newaxis] ) )
    # print( np.shape( ( np.linalg.norm( arrWeightedTest, axis=-1 ) * np.linalg.norm( centroid, axis=-1 ) )[:, np.newaxis] ))
    similarity = np.sum( arrWeightedTest * centroid, axis=-1 )[:, np.newaxis] / ( np.linalg.norm( arrWeightedTest, axis=-1 ) * np.linalg.norm( centroid, axis=-1 ) )[:, np.newaxis]

    print( np.shape( similarity ) )
    pred = np.argmax( similarity, axis=-1 )
    # print( pred )
    pred = [ ind[0] for ind in pred ]
    print( reverseMappings )
    # dictLabelToFile = { 1:'a', 2:'b', 3:'c', 4:'d', 5:'e' }
    # dictLabelToFile = { 1:'a', 2:'b', 3:'c', 4:'d', 5:'e', 6:'f' }
    label = [ list(dictLabelToFile.keys())[i] for i in pred ]
    # print( label ) # Get class label
    names = [ reverseMappings[i+1] for i in pred ]
    # print( names )

    with open( os.path.join( os.getcwd(), "../", "preds.txt" ), 'w' ) as f:
        for i in range( len( listTestFile ) ):
            f.write( listTestFile[i] + " " + names[i] + "\n" )