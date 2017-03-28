nf = open('FilteredEntityMention.txt', 'w+')

f = open('../data/KBP16/en/diffbot_linked.txt', 'r')
lines = f.readlines()

for line in lines:
    tokens = line.split()

    if len(tokens) < 4:
        break

    confValue = tokens[len(tokens) - 2]

    if float(confValue) > 0.65:
        nf.write(line)

f.close()
nf.close()
