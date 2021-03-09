from tf_idf import TF_IDF
import os


if __name__ == "__main__":
    
    model = TF_IDF()    

    ############ Train Step ############
    if __debug__:
        # fileTrainList = "corpus1_train.labels"
        fileTrainList = "../train.txt"
    else:
        fileTrainList = input( "Please input the path to the file containing the list of labeled training files: " )

    model.train( fileTrainList )

    ############ Test Step ############
    if __debug__:
        # fileTestList = "corpus1_test.list"
        fileTestList = "../test.txt"
        fileOutput = "../preds.txt"
    else:
        fileTestList = input( "Please input the path to the file containing the list of testing files: " )
        fileOutput = input( "Please input the path to the file to save predictions to: " )
        
    model.test( fileTestList, fileOutput )
