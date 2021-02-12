import re

# match the play song by artist type of utterance
groups = re.match('(playing|play|starting) (\w.+) by (\w.+) (on amazon music|on apple music|from spotify)', 'playing our song by taylor swift on amazon music')