import numpy as np
import os
from classify import Classifier


class TF_IDF( Classifier ):
    
    def __init__( self ):
        pass

    def calcSimilarity( self, inputs, centroids ):

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

    def calcIDF( self, fileCounts, numTotalWords ):

        '''
        Calculates log Inverse Document Frequency for each word
        Defined as:
            log( Total # of documents / # of documents that contain the word )
        '''
        self.idf = np.zeros( (numTotalWords,1) )
        for i in range( self.numDocs ):
            self.idf[ np.array(list( fileCounts[i].keys() ))-1 ] += 1
        self.idf = np.log10( self.numDocs / self.idf )

    def calcWeights( self, fileCounts ):

        # Calculate weighted tf-idf value
        weighted = []
        for i in range( len( fileCounts ) ):
            weighted.append( { k:( np.log10(v+1) * self.idf[k-1] ).item() for (k,v) in fileCounts[i].items() } )

        return weighted

    def train( self, fileTrainList ):

        # Get the word counts for training documents
        self.loadTrain( fileTrainList )

        # Calculate the IDF
        self.calcIDF( self.trainCounts, self.numTotalWords )

        # Calculate the weighted word counts
        weighted = self.calcWeights( self.trainCounts ) 

        # Convert dict of weighted word counts to array
        arrWeighted = np.zeros( (self.numDocs,self.numTotalWords) )
        for i in range( self.numDocs ):
            for key in weighted[i].keys():
                arrWeighted[i][key-1] = weighted[i][key]

        # Calculate centroid of each class
        self.centroids = np.zeros( (len( self.dictLabelToFile.keys() ), self.numTotalWords) )
        for key in self.dictLabelToFile.keys():

            listDocs = self.dictLabelToFile[key]
            self.centroids[key-1] = np.average( arrWeighted[ listDocs ], axis=0 )

    def evaluate( self, testWeightedCounts, fileOutput ):

        # Calculate cosine similarity
        similarity = self.calcSimilarity( testWeightedCounts, self.centroids )

        # Maximize the similarity and extract predictions from nested list
        # E.g. [ [x],[y],[z] ] -> [ x,y,z ]
        pred = np.argmax( similarity, axis=-1 )
        pred = [ ind[0] for ind in pred ]
        names = [ self.reverseMappings[i+1] for i in pred ]

        # Save predictions to a file
        with open( os.path.join( os.getcwd(), fileOutput ), 'w' ) as f:
            for i in range( len( self.testNames ) ):
                f.write( self.testNames[i] + " " + names[i] + "\n" )

    def test( self, fileTestList, fileOutput ):

        # Get the word counts for testing documents
        self.loadTest( fileTestList )

        # Calculate the weighted word counts
        weightedTest = self.calcWeights( self.testCounts ) 

        # Convert to array
        arrWeightedTest = np.zeros( ( len(weightedTest), self.numTotalWords ) )
        for i in range( len(self.testCounts) ):
            for key in weightedTest[i].keys():
                arrWeightedTest[i][key-1] = weightedTest[i][key]

        # Add a dimension for broadcasting
        arrWeightedTest = arrWeightedTest[:, np.newaxis, :]

        # Get predictions and write to file
        self.evaluate( arrWeightedTest, fileOutput )
