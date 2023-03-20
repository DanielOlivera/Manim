from pytube import *
from pytube.cli import *
link = str(input("Link: "))
yt = YouTube(link, on_progress_callback=on_progress)
stream = yt.streams.get_highest_resolution()
stream.download()
print("Download Successful")