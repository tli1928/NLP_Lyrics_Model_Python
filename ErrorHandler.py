"""
Ben Ecsedy, Jack Krolik, Teng Li, Joey Scolponeti
DS3500
Homework 3
2/27/2023
"""


class ParserErrors(Exception):
    """ Class for Parser Errors inherited from python exceptions class"""
    pass


class FileNotFound(ParserErrors):

    def __init__(self, filename):
        """ For errors where the file is not found

        Args:
            filename (string): filename of the song
        Returns (string): an error to the user
        """
        super().__init__("The file " + "'" + str(filename) + "'" + " was not found, " \
                         "please ensure it is in your directory path")


class Index(ParserErrors):
    """ For errors where the file is not found
        Args:
            filename (string): filename of the song

        Returns (string): an error to the user


    """

    def __init__(self, filename):
        super().__init__("The parser is having trouble with the file " + "'" + str(filename) + "' " + \
                         "please ensure it is .lrc file when using the read_lrc parser")
