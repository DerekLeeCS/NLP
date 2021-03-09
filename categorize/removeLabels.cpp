#include <string>
#include <vector>
#include <iostream>
#include <fstream>


using std::string;
using std::vector;
using std::cout;
using std::cin;
using std::cerr;
using std::endl;
using std::ifstream;
using std::ofstream;


template <class T>
void checkValidStream( T &, string );

int main() {

    string fileIn, fileOut;
    vector<string> fileContents;

    cout << "Enter the name of the Input file: ";
    cin >> fileIn;
    ifstream input( fileIn );
    checkValidStream( input, fileIn );

    cout << "Enter the name of the Output file: ";
    cin >> fileOut;
    ofstream outputTrain( fileOut );
    checkValidStream( outputTrain, fileOut );

    // Counts number of lines
    string line;
    std::size_t loc;
    while ( std::getline(input,line) ) {

        if ( line.length() > 0 ) {

            loc = line.find(' ');
            outputTrain << line.substr(0,loc) << '\n';

        }

    }
    input.close();

    return 0;

}

template <class T>
void checkValidStream( T &stream, string fileName ) {

    if ( !stream ) {

        cerr << "Error: could not open " << fileName << "\n";
        exit( EXIT_FAILURE );

    }

}
