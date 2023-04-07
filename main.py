from NLP import NaturalLanguage
import os

def main():
    """
    Calls NLP library to produce visualizations

    """

    # initializing file name list
    filename_list = []

    # get all file names from songs folder
    for file in os.listdir('Songs'):
        song = os.path.join('Songs', file)
        filename_list.append(song)

    # clean song names to extract title for labels
    labels = [x[6:].strip('.lrc') for x in filename_list]

    # generate instance of nlp object
    nlp = NaturalLanguage(filename_list, labels=labels, parser='read_lrc')

    # generate repetition plot
    nlp.plot_repetition(filenames=filename_list, labels=labels)

    # generate sentiment plot
    nlp.plot_sentiment(filenames=filename_list, labels=labels)

    # generate sankey plot
    nlp.wordcount_sankey(num_words=10)



main()

