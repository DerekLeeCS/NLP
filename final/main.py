import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from loadFile import loadList, tokenizeFile, createMappings


# Preprocessing
# text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
# preprocessor = hub.KerasLayer(
#     "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
# encoder_inputs = preprocessor(text_input)


if __name__ == "__main__": 

    ############ Train Step ############
    if __debug__:
        # fileTrainList = "corpus1_train.labels"
        fileTrainList = "../train.txt"
    else:
        fileTrainList = input( "Please input the name of the file containing the list of labeled training files.\n" )

    # Load the training files
    listTrainFiles, listTrainLabels = loadList( fileTrainList )
    mappingsLabels, reverseMappings = createMappings( listTrainLabels )
    listTrainLabels = list( map( mappingsLabels.get, listTrainLabels ) )

    # Encode the text
    dsText, encoder = tokenizeFile( listTrainFiles, listTrainLabels )

    # Create the model
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=64,
            # Use masking to handle the variable sequence lengths
            mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),            ########## Change to RNN
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    ############ Test Step ############
    if __debug__:
        # fileTestList = "corpus1_test.list"
        fileTestList = "../test.txt"
        fileOutput = "../preds.txt"
    else:
        fileTestList = input( "Please input the name of the file containing the list of testing files.\n" )
        fileOutput = input( "Please input the name of the file to save predictions to.\n" )
        
    


# BERT
# encoder = hub.KerasLayer(
#     "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3",
#     trainable=True)
# outputs = encoder(encoder_inputs)
# pooled_output = outputs["pooled_output"]      # [batch_size, 768].
# sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].