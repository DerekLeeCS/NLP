from tf_idf import TF_IDF
import os


if __name__ == "__main__":
    
    model = TF_IDF()    

    print( "Please input all filepaths relative to the current working directory:\n", os.getcwd(), sep='', end='\n\n' ) ################ REMOVE LATER

    ############ Train Step ############
    if __debug__:
        # fileTrainList = "corpus1_train.labels"
        fileTrainList = "../train.txt"
    else:
        fileTrainList = input( "Please input the name of the file containing the list of labeled training files.\n" )

    model.train( fileTrainList )

    ############ Test Step ############
    if __debug__:
        # fileTestList = "corpus1_test.list"
        fileTestList = "../test.txt"
        fileOutput = "../preds.txt"
    else:
        fileTestList = input( "Please input the name of the file containing the list of testing files.\n" )
        fileOutput = input( "Please input the name of the file to save predictions to.\n" )
        
    model.test( fileTestList, fileOutput )
