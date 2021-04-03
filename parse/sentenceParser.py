from __future__  import annotations
from collections import defaultdict
from typing import List, DefaultDict, Tuple, Union

### Type Declarations ###

# Stores location of component POS and name of POS
# Used for backtracking
Component = Tuple[ int,int,int,str,str ]

# 2D array created by CKY algorithm
ArrCKY = List[ List[set] ]

# Maps a POS to its component POS
# Used for backtracking
ArrReverseCKY = List[ List[ DefaultDict[ str, List[Component] ] ] ]

# ParseTree is recursive
# Base type is ( POS, word )
ParseTree = Union[ Tuple[ str, 'ParseTree', 'ParseTree' ], Tuple[ str, str ] ]

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
    def parse( self, words: List[str] ) -> Tuple( ArrCKY, ArrReverseCKY ):
        
        # Count number of words in sentence
        n = len( words )
        
        # Create n by n matrix of empty sets
        matrix = [ [ set() for _ in range(n) ] for _ in range(n) ]

        # Used for backtracking to determine component constituents
        constituents = [ [ defaultdict( list ) for _ in range(n) ] for _ in range(n) ]    

        for j in range(1,n):

            for A in self.reverseTerminals[ words[j] ]:
                matrix[j-1][j] = matrix[j-1][j].union([A])
                constituents[j-1][j][A] = None

            for i in reversed( range(j) ):
                for k in range( i, j ):
                    for B in matrix[i][k]:
                        for C in matrix[k][j]:
                            for A in self.reverseNonTerminals[ tuple( [B,C] ) ]:
                                matrix[i][j] = matrix[i][j].union([A])
                                constituents[i][j][A].append( tuple( [i,k,j,B,C] ) )

        return matrix, constituents

    def getParses( self, constituents: ArrReverseCKY, words: List[str], row: int, col: int, targetPOS: str ) -> List[ ParseTree ]: 

        # Reached terminal
        if row == col-1:
            return [( targetPOS, words[col] )] \
                if targetPOS in constituents[row][col].keys() \
                else []

        # Reached non-terminal
        # Credit to Jonathan Lam for coming up with this crazy nested list comprehension
        return [ ( targetPOS, parse_i, parse_j ) \
                    for i,k,j,B,C in constituents[row][col][targetPOS] \
                    for parse_i in self.getParses( constituents, words, i, k, B ) \
                    for parse_j in self.getParses( constituents, words, k, j, C ) ]

    # Prints a specific parse of the sentence with indents
    def formatParse( self, listParseTrees: List[ ParseTree ], depth: int ) -> str:

        parsedSentence = ""
        indent = "\t"*depth

        if len( listParseTrees ) == 2:
            return indent + "[" + listParseTrees[0] + " " + listParseTrees[1] + "]"
        else:
            parsedSentence += indent + "[" + listParseTrees[0] + "\n"
            parsedSentence += self.formatParse( listParseTrees[1], depth+1 ) + "\n"
            parsedSentence += self.formatParse( listParseTrees[2], depth+1 ) + "\n" + indent + "]"

        return parsedSentence

    # Returns a formatted string in bracket notation
    # Used after formatParse()
    def stripFormat( self, formattedStr: str ) -> str:
        return formattedStr.replace("\t", "").replace("\n", " ").replace(" ]", "]")

    # Displays all parsed sentences
    def display( self, constituents: ArrReverseCKY, words: List[str], boolTrees: bool ) -> None:
        
        n = len( words )
        
        # Get all valid parses
        listParseTrees = self.getParses( constituents, words, 0, n-1, 'S' )

        if len( listParseTrees ) == 0:
            print( "NO VALID PARSES" )
            return

        # Loop through all valid parses and print
        for i in range( len( listParseTrees ) ):

            print( "Parse #", i+1, ":", sep="" )
            formattedStr = self.formatParse( listParseTrees[i], 0 )
            print( self.stripFormat( formattedStr ) )

            if boolTrees:
                print()
                print( formattedStr )

        print()
        print( "Number of valid parses:", len( listParseTrees ) )
