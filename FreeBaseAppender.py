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

fileFreeBaseMidTypeSample = open('../data/KBP16/freebase-mid-type_sample.map', 'r')
lineFreeBaseMidTypeSample = fileFreeBaseMidTypeSample.readlines()

sampleDict = {}

for line in lineFreeBaseMidTypeSample:
    tokens = line.split()

    if len(tokens) < 2:
        continue

    freebaseValue = tokens[0]
    freebaseType = tokens[1]

    freebaseValue = freebaseValue[1:-1]
    freebaseType = freebaseType[1:-1]

    freebaseTypeConverted = freebaseType.replace('http://rdf.freebase.com/ns', '')
    freebaseTypeConverted = freebaseTypeConverted.replace('.', '/')

    if freebaseTypeConverted in typeMapDict:
        sampleDict[freebaseValue] = typeMapDict[freebaseTypeConverted]
    else:
        sampleDict[freebaseValue] = ''

fileFreebaseLink = open('../data/KBP16/freebase_links.nt', 'r')
linesFreebase = fileFreebaseLink.readlines()

linkDict = {}

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
    else:
        linkDict[dbpediaValue] = (freebaseValue, '')


fileMentions = open('FilteredEntityMention.txt', 'r')
fileFreebase = open('FilteredEntityMentionWithFreebase.txt', 'w+')
fileFreebaseUrl = open('FreebaseUrl.txt', 'w+')

mentionLines = fileMentions.readlines()

for line in mentionLines:
    tokens = line.split()

    if len(tokens) < 3:
        continue

    dbpediaUrl = tokens[len(tokens) - 3]

    if dbpediaUrl in linkDict:

        (freebaseUrl, cattypes) = linkDict[dbpediaUrl]
        fileFreebaseUrl.write(freebaseUrl + "\t" + line)

        if len(cattypes) > 1:
            print cattypes
        freebaseLine = line[:len(line)-1] + "  \t" + freebaseUrl + "  \t" + cattypes + "\t" + line[len(line)-1]
        fileFreebase.write(freebaseLine)
