"""
Contains a class
"""

class AudioEvent:
    """
    Class for storing audio events in a csv file
    """
    def __init__(self, time, note, volume):
        self.time = time
        self.note = note
        self.volume = volume

    def to_csv_line(self):
        """
        Returns a csv line
        """
        return f'{self.time:.4f},{self.note},{self.volume:.4f}'

    @staticmethod
    def from_csv_line(line):
        """
        Constructs an AudioEvent from a csv line
        """
        values = line.split(',')
        return AudioEvent(
            float(values[0]),
            int(values[1]),
            float(values[2])
        )
