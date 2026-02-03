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

- C:\Users\shado\Year3Projects\FYP\src\dataset\bird_data\clips_v1\species\corvus_corax\XC1027629__s27000__e30000.wav you can hear the wings flapping
- C:\Users\shado\Year3Projects\FYP\src\dataset\bird_data\clips\non_bird\apus_apus\XC1059011__s1000__e4000.wav Caught a goddamn sheep



=== Downloading pica pica ===
1 / 250
2 / 250
3 / 250
4 / 250
5 / 250
6 / 250
7 / 250
8 / 250
9 / 250
10 / 250
Failed pica pica: HTTPSConnectionPool(host='xeno-canto.org', port=443): Max retries exceeded with url: /767200/download (Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x000002229FD5F0D0>, 'Connection to xeno-canto.org timed out. (connect timeout=60)'))


143 / 250
Failed phylloscopus collybita: 500 Server Error: Internal Server Error for url: https://xeno-canto.org/684749/download


82 / 250
Failed sylvia atricapilla: 404 Client Error: Not Found for url: https://xeno-canto.org/1054532/download

BirdNet uses a diff scientific name for Jackdaw than xeno canto





Meeting Agenda
- Last time had data exploration (stats) finished
- Over Christmas aimed to get first model attempt finished
- Completed:
    - Data gathering, help model generalise, ensured good quality audio
    - Fixed size clipping script, used birdnet api to verify high energy segments matched labelled species
    - Added non-bird class
    - Finalised on device flow while designing the data
    - Researched spectrograms and the Max78000
    - Current spectrograms are quite small, might have to move to max78002 later
    - Now I need to design model (probably ResNet) and train on cloud

TODO
- Train model
- Measure accuracies
- Repeat until i get good model


