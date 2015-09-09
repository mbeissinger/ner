import re
import os
from opendeep.utils.file_ops import find_files

def fullClean(s):
        if (s is None):
            return ''
        else:
            s = s.decode('string_escape', 'ignore')
            s = s.decode('unicode_escape', 'ignore')
            s = s.replace(u"\u2018", "'")\
                 .replace(u"\u2019", "'")\
                 .replace(u"\u201c", '"')\
                 .replace(u"\u201d", '"')\
                 .replace(u"\u00a0", ' ')
            s = s.replace('\n', '. ')
            s = s.encode('ascii', 'ignore')
            s = s.replace('\\,', ',').replace('\t', ' ').replace('\/', '/')
            s = s.decode('utf-8', 'ignore')
            s = re.sub('\.?(\.\s)+', '. ', s)
            s = re.sub('\s+', ' ', s).strip()
            if s.startswith('"'):
                s = s[1:]
            if s.endswith('"'):
                s = s[:-1]
            s = s.strip()

            return s

def main():
    for fpath in find_files(os.path.join('data', 'tokenized')):
        base = os.path.basename(fpath)
        rawpath = os.path.join('data', 'raw', base)

        with open(rawpath, 'r') as f:
            rawtext = fullClean(f.read()).lower()

        with open(fpath, 'r') as f:
            tokenized = []
            for line in f:
                word = line.strip().split('\t')[0].lower()
                label = line.strip().split('\t')[1]
                if word == "``":
                    word = '"'
                elif word == "''":
                    word = '"'
                tokenized.append((word, label))

            if len(tokenized)>0 and tokenized[0][0] == '"':
                tokenized.pop(0)
            if len(tokenized)>0 and tokenized[-1][0] == '"':
                tokenized.pop(-1)

        prev_label = None
        for char in rawtext:


        print rawtext
        print tokenized

        break


if __name__ == "__main__":
    main()
