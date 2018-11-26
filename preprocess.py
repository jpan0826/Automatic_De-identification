import glob
import xml.etree.ElementTree as ET
import nltk

train_dir = '../datasets/train/'
test_dir = '../datasets/test/testing-PHI-Gold-fixed/'

def parse_xml(dir):
    data = []
    for filename in glob.iglob(dir+'**/*.xml', recursive=True):
        dict = {}
        tree = ET.parse(filename)
        root = tree.getroot()
        for log in root.iter("TEXT"):
            text = log.text
            dict['text'] = text
            dict['tokens'] = nltk.word_tokenize(text)
        for child in root:
            if child.tag == 'TAGS':
                tags = [element.attrib for element in child]
                dict['tags'] = tags
        data.append(dict)
    return data
