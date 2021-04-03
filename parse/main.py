from sentenceParser import CKY


if __name__ == "__main__":

    fileCFG = input( "Please enter the name of the text file specifying a Context-Free Grammar in Chomsky normal form: " )

    # Initialize CKY class
    model = CKY()
    model.processCFG( fileCFG )
    print( "Loading grammar..." )

    if ( input( "Do you want textual parse tress to be displayed (y/n)?: " ) == 'y' ):
        boolTrees = True
    else:
        boolTrees = False

    # Keep parsing sentences until user stops the program
    while True:

        userInput = input( "Please enter a sentence to parse: " )

        if userInput == "quit":
            break
        
        # Split the sentence into a list of words and add a 'fencepost' at the beginning
        words = userInput.split()
        words = [ "_" ] + words

        # Parse the sentence and print the potential parses
        _, constituents = model.parse( words )
        model.display( constituents, words, boolTrees )

        print()