from collections import defaultdict
from typing import List, DefaultDict, Tuple

class CKY:

    def __init__( self ) -> None:
        
        # Process grammar into rules for terminals and non-terminals
        # If key has not been seen before, value is an empty list
        self.reverseTerminals = defaultdict( list )
        self.reverseNonTerminals = defaultdict( list )

    def processCFG( self, fileName: str ) -> None:
        
        with open( fileName, 'r' ) as f:

            for rule in f:
                
                symbols = rule.split()

                # A -> B C
                if len( symbols ) == 4:
                    A, _, B, C = symbols
                    self.reverseNonTerminals[ tuple( [B,C] ) ].append(A)

                # A -> w
                elif len( symbols ) == 3:
                    A, _, w = symbols
                    self.reverseTerminals[w].append( A )

                # Invalid, should never happen
                else:
                    print( "Invalid Rule:", rule )
                    exit(-1)


    # CKY Algorithm
    def parse( self, words: List[str] ) -> List[ List[set] ]:
        
        # Count number of words in sentence
        n = len( words )
        
        # Create n+1 by n+1 matrix of empty sets
        matrix = [ [ set() for _ in range(n) ] for _ in range(n) ]  # possibly n+1

        # Used for backtracking to determine component constituents
        constituents = [ [ defaultdict( list ) for _ in range(n) ] for _ in range(n) ]    

        for j in range(1,n):

            for A in self.reverseTerminals[ words[j] ]:
                matrix[j-1][j] = matrix[j-1][j].union([A])
                constituents[j-1][j][A] = None

            for i in reversed( range(j-1) ):
                for k in range( i+1, j ):
                    for B in matrix[i][k]:
                        for C in matrix[k][j]:
                            for A in self.reverseNonTerminals[ tuple( [B,C] ) ]:
                                matrix[i][j] = matrix[i][j].union([A])
                                constituents[i][j][A].append( tuple( [i,k,j,B,C] ) )


        return matrix, constituents

    def getParses( self, constituents: List[ List[ DefaultDict[str,List] ] ], words: List[str], row: int, col: int, targetPOS: str ) -> List: 

        # Reached terminal
        if row == col-1:
            return [ [( targetPOS, words[col] )] ] \
                if targetPOS in constituents[row][col].keys() \
                else []

        # Reached non-terminal
        listParseTrees = []
        for i,k,j,B,C in constituents[row][col][targetPOS]:
            parseLeft = self.getParses( constituents, words, i, k, B )
            parseRight = self.getParses( constituents, words, k, j, C )
            listParseTrees.append( [ ( targetPOS, parse_i, parse_j ) \
                                for parse_i in parseLeft \
                                for parse_j in parseRight ] )

        return listParseTrees

    # Prints a specific parse of the sentence with indents
    def printParse( self, listParseTrees, depth ):

        parsedSentence = ""

        if len( listParseTrees ) == 2:
            return "\t"*depth + "[" + listParseTrees[0] + " " + listParseTrees[1] + "] "
        else:
            parsedSentence += "\t"*depth + "[" + listParseTrees[0] + "\n"
            parsedSentence +=  self.printParse( listParseTrees[1][0], depth+1 ) + "\n"
            parsedSentence +=  self.printParse( listParseTrees[2][0], depth+1 ) + "]"

        return parsedSentence

    # Displays a calculated parse table
    def display( self, constituents: List[ List[ DefaultDict[str,List] ] ], words: List[str] ) -> None:
        
        n = len( words )

        # Get all valid parses
        listParseTrees = self.getParses( constituents, words, 0, n-1, 'S' )

        if len( listParseTrees ) == 0:
            print( "NO VALID PARSES" )
            return

        # Loop through all valid parses and print
        for i in range( len( listParseTrees ) ):
            finalStr = self.printParse( listParseTrees[i][0], 0 )
            print( finalStr )
