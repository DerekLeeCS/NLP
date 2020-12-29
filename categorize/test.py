testFile = "./corpus1/test/33120.article"
    testCount = defaultdict( lambda:0 )

    with open( os.path.join( os.getcwd(), testFile ), 'r' ) as f:

        fContents = f.read()

        for word in word_tokenize( fContents ):
        
            # If word has not been previously seen, count is 0
            testCount[word] += 1

    # Ignores a word if has not been previously seen
    # B/c unknown words map to an idf value of 0
    testCount = { mappingsWords[k]:v for (k,v) in testCount.items() if k in mappingsWords }
    weightedTest = { k:( np.log10(v+1) * idf[k-1] ).item() for (k,v) in testCount.items() }

    arrWeightedTest = np.zeros( (1,totalWords) )
    for key in weightedTest.keys():
        arrWeightedTest[0][key-1] = weightedTest[key]

    # print( np.shape( np.linalg.norm( centroid, axis=-1 ) ) )
    # print( np.shape(  np.linalg.norm( arrWeightedTest, axis=-1 )) )
    # print( np.shape( arrWeightedTest * centroid ) )
    print( np.shape( np.sum( arrWeightedTest * centroid, axis=-1 )[:, np.newaxis] ) )
    print( np.shape( ( np.linalg.norm( arrWeightedTest, axis=-1 ) * np.linalg.norm( centroid, axis=-1 ) )[:, np.newaxis] ))
    similarity = np.sum( arrWeightedTest * centroid, axis=-1 )[:, np.newaxis] / ( np.linalg.norm( arrWeightedTest, axis=-1 ) * np.linalg.norm( centroid, axis=-1 ) )[:, np.newaxis]
    
    print( similarity )
    pred = np.argmax( similarity )
    print( pred )
    label = list(dictLabelToFile.keys())[pred]
    print( label ) # Get class label
    print( reverseMappings[ label ])