from nltk.tokenize import word_tokenize
from collections import defaultdict
from typing import List, Dict, DefaultDict
import os


class Utility:

    '''Base class containing useful functions'''

    def __init__( self ):
        pass

    # Used to map a string label to an int
    def mapStrToInt( self, listString: List[str] ) -> Dict[str,int]:

        '''Returns a dict to map each unique string to an int'''

        mappings = { label:num+1 for (num,label) in enumerate( sorted( set(listString) ) ) }
        return mappings

    # Create mappings given a list of labels
    def createMappings( self, listLabels: List[str] ) -> tuple( [ Dict[str,int], Dict[int,str] ] ):

        # Get dict for mapping 
        mappingsLabels = self.mapStrToInt( listLabels )
        reverseMappings = { v:k for (k,v) in mappingsLabels.items() }

        return mappingsLabels, reverseMappings

    # Load the list of files
    def loadList( self, fileList: List[str] ) -> tuple( [ List[str], List[str], int ] ):

        listFiles = []
        listLabels = []
        N = 0

        # Read file to get list of files & labels
        with open( os.path.join( os.getcwd(), fileList ), 'r' ) as f:
            
            for line in f:

                path, label = line.split()
                listFiles.append( path )
                listLabels.append( label )
                N += 1

        return listFiles, listLabels, N

    # Used to get counts of unique words in each file
    def countWords( self, listFiles: List[str] ) -> tuple( [ List[ DefaultDict[str,int] ], DefaultDict[str,int] ] ):

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


class Classifier( Utility ):

    '''Base class for classifier methods'''

    def __init__( self ):
        pass

    # Get word counts for training documents
    def loadTrain( self, fileTrainList: List[str] ) -> None:

        listTrainFiles, listTrainLabels, self.numDocs = self.loadList( fileTrainList )

        # Convert labels from str -> int
        self.mappingsLabels, self.reverseMappings = self.createMappings( listTrainLabels )
        listTrainLabels = list( map( self.mappingsLabels.get, listTrainLabels ) )

        # Map labels -> documents
        # Groups documents by class
        self.dictLabelToFile = defaultdict( list )
        for i in range( len( listTrainLabels ) ):
            self.dictLabelToFile[ listTrainLabels[i] ].append(i) 

        # Get word counts for each training file
        self.trainCounts, self.totalCounts = self.countWords( listTrainFiles )

        # Get mappings for words from str -> int
        self.mappingsWords = self.mapStrToInt( self.totalCounts.keys() )

        # Map words 
        self.totalCounts = { self.mappingsWords[k]:v for (k,v) in self.totalCounts.items() }
        for i in range( len(self.trainCounts) ):
            self.trainCounts[i] = { self.mappingsWords[k]:v for (k,v) in self.trainCounts[i].items() }
        self.numTotalWords = len( self.totalCounts.keys() )

    # Get word counts for testing documents
    def loadTest( self, fileTestList: List[str] ) -> None:

        # Save file names for writing results
        self.testNames, _, _ = self.loadList( fileTestList )

        # Get word counts for each testing file
        self.testCounts, _ = self.countWords( self.testNames )

        # Map words 
        # Ignores a word if has not been previously seen
        for i in range( len(self.testCounts) ):
            self.testCounts[i] = { self.mappingsWords[k]:v for (k,v) in self.testCounts[i].items() if k in self.mappingsWords }