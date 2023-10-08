import re
from youtube_transcript_api import YouTubeTranscriptApi
from summary import generate_summary
from grammar import grammify

link = input("LINK = ")
video_id = link.split('=')[1]
print(video_id)
# Get the transcript
transcript = YouTubeTranscriptApi.get_transcript(video_id)

# Clean the transcript
cleaned_transcript = ""
for line in transcript:
    text = line["text"]
    # Remove speaker tags (e.g. [Music])
    text = re.sub(r'\[.*?\]', '', text)
    # Remove timestamps (e.g. 00:01:23.456)
    text = re.sub(r'\d{1,2}:\d{2}:\d{2}\.\d{3}', '', text)
    # Remove any remaining brackets and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    cleaned_transcript += text + "\n"

print(cleaned_transcript)

x = generate_summary(cleaned_transcript)
grammify(x)
