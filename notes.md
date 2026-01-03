# Notes

## Intrusive Thoughts
- Are Plovers fitting my woodland dataset? 
- I'm going to keep gulls because I'm a big gull fan.
- I fucking love ravens
- Really funny that the rarer the bird, the more audio you can find.
    - Only 200 rock dove recordings? 
- Holy shit there are so many warblers
- I'm actually laughing so much at these scientific names:
    - Troglodytes troglodytes
    - Pica pica
- Not many Bluethoats spotted but im keeping them cause they look cool 
- "The New World warblers are a group of small, often colourful, passerine birds restricted to the New World." 
    - What is the new world?!!! Is this Agartha?? 
- https://xeno-canto.org/1045711 This one goes really hard



# Thoughts
- How am I going to check if there is a bird? 
    - Do I have the 50 bird classes + unknown / not bird
    - Or do I have an inital model that checks if bird or not? 
    Flow would be:
    - Convert audio buffer to spectrogram
    - Pass spectrogram to init model
    - If bird: pass to classification model
    - Else: drop it? 
    - Init model version would be a lot of extra work.. and I'm not sure how to run two models on the MAX78000. 
    - Maybe use two MAX78000's? Or one 78000 and one 78002?
    - One model would be more simple, but I would need data for birds + data for unknown/not bird

- "Long recordings were split into 5 s chunks, and only the chunks containing bird sounds were kept for training"
    - How on earth did they do that? 
    - Maybe just capture the highest energy 5s chunk from each audio file? Less data but more chance it captures the important part of the calls?
    - Actually what if I use a bigger model, and on each 3s clip, keep the ones that the model can correctly classify (up to certain confidence)
    - Many of these paper models are very good at classifying, but I am pretty sure they are doing it in controlled environments. My model needs to be able to classify bird calls well, and in real time. DuSAFNet seems to have a very good check for detecting calls vs silence, but idk if it is practical for my stuff.

# Bird@Edge Notes
- The ESP32 microphone stuff will be useful for my project. I can do the same thing with the ESP32 I stole from ISE.
- Take a look at BirdNetLite
- Does not mention data much


# Picking CSV data
- I want to store basic info about each recording in a csv file, so like sci name, id, file location, xeno canto url.
- I also want to store filtering information (e.g. country, also, animal seen)
- I am going to store date/time of recording and maybe try encode it into predictions. It would be big help if one of the hardware units could encode the date and time into the data to predict, because some birds only visit ireland in the winter, or sing in certain months. 