import json
from collections import Counter
from ErrorHandler import *



def read_lrc(filename):
    """
    Reads a lyrical file
    :param filename (string): name of the lrc file to read
    :return (string): song as string
    """
    # file not found error
    # file is not of lrc type
        # read the file
    try:
        with open(filename) as f:
            contents = f.read()

    except FileNotFoundError:
        raise FileNotFound(filename)

    # get the contents of the file as a list of lyrics
    contents = contents.split("\n")

    song = ''
    # for every lyric in contents add it to the song string
    for line in contents:
        try:
            line = line.split(']')
            lyric = line[1]
        except IndexError:
            raise Index(filename)

        song = song.strip()
        song += '\n' + lyric

    return song.strip()










