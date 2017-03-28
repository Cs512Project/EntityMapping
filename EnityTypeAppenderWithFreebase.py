import os

fileTypeMap = open('../data/KBP16/type.map', 'r')
linesType = fileTypeMap.readlines()

typeMapDict = {}

for line in linesType:
    tokens = line.split()

    if len(tokens) < 2:
        continue

    freebasetype = tokens[0]
    cattype = tokens[1]

    if freebasetype in typeMapDict:
        typeMapDict[freebasetype] += cattype
    else:
        typeMapDict[freebasetype] = cattype

fileFreeBaseMidTypeSample = open('../data/KBP16/freebase-mid-type_sample_2.map', 'r')
fileMissingTypeMapping = open('MissingTypeMapping.txt', 'w+')   # output file
lineFreeBaseMidTypeSample = fileFreeBaseMidTypeSample.readlines()

sampleDict = {}

countTypeMap = 0
countTypeNotMap = 0

for line in lineFreeBaseMidTypeSample:
    tokens = line.split()

    if len(tokens) < 2:
        continue

    freebaseValue = tokens[0]
    freebaseType = tokens[1]

    # freebaseValue = freebaseValue[0:-1]
    # freebaseType = freebaseType[0:-1]

    # freebaseTypeConverted = freebaseType.replace('http://rdf.freebase.com/ns', '')
    freebaseTypeConverted = '/' + freebaseType.replace('.', '/')
    freebaseValueConverted = "http://rdf.freebase.com/ns/" + freebaseValue

    if freebaseTypeConverted in typeMapDict:
        sampleDict[freebaseValueConverted] = typeMapDict[freebaseTypeConverted]
        countTypeMap += 1
    else:
        sampleDict[freebaseValueConverted] = ''
        fileMissingTypeMapping.write(freebaseValueConverted + "\t" + freebaseTypeConverted + os.linesep)
        countTypeNotMap += 1

print countTypeMap
print countTypeNotMap

fileFreebaseLink = open('../data/KBP16/freebase_links.nt', 'r')
linesFreebase = fileFreebaseLink.readlines()

linkDict = {}

countFbMap = 0
countFbNotMap = 0

for line in linesFreebase:
    tokens = line.split()

    if len(tokens) < 3:
        break

    dbpediaValue = tokens[0]
    freebaseValue = tokens[len(tokens) - 2]

    dbpediaValue = dbpediaValue[1:-1]
    freebaseValue = freebaseValue[1:-1]

    if freebaseValue in sampleDict:
        linkDict[dbpediaValue] = (freebaseValue, sampleDict[freebaseValue])
        countFbMap += 1
    else:
        linkDict[dbpediaValue] = (freebaseValue, '')
        countFbNotMap += 1

print countFbMap
print countFbNotMap

fileMentions = open('FilteredEntityMention.txt', 'r')
fileFreebase = open('FilteredEntityMentionWithFreebase.txt', 'w+')  # output file
fileFreebaseUrl = open('FreebaseUrlType.txt', 'w+')                 # output file

mentionLines = fileMentions.readlines()

for line in mentionLines:
    tokens = line.split()

    if len(tokens) < 3:
        continue

    dbpediaUrl = tokens[len(tokens) - 3]

    if dbpediaUrl in linkDict:

        (freebaseUrl, cattypes) = linkDict[dbpediaUrl]
        # fileFreebaseUrl.write(freebaseUrl + "\t" + line)

        if len(cattypes) > 1:
            print cattypes

        freebaseLine = line[:len(line)-1] + "  \t" + freebaseUrl + "  \t" + cattypes + "\t" + line[len(line)-1]
        fileFreebase.write(freebaseLine)

        if len(cattypes) > 1:
            print cattypes
            fileFreebaseUrl.write(freebaseLine)

print "finished"
