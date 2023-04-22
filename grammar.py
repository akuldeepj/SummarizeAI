from gingerit.gingerit import GingerIt
# Split the text into chunks of 200 words each
def grammify(text):

    words = text.split()
    chunks = [' '.join(words[i:i+70]) for i in range(0, len(words), 70)]
    parser = GingerIt()

    for chunk in chunks:
        x = parser.parse(chunk)
        print(x['result'],end=' ')


