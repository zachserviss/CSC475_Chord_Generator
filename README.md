# CSC475_Chord_Generator
Repo for CSC475 Project

# Notes
* I have included a env.txt file with the dependencies I'm using to run the notebook, if you need to replicate.
* Note that the notebook uses a `data` folder to access the wav and the csv files.
* I have not included the data folder as it's pretty big. It can be downloaded from [here](https://drive.google.com/drive/folders/1IAxn4qTwzk0Ri6Zpev5--_j8RN3YzCke?usp=sharing)
* Put the `data` folder in the root directory of this repo.
* I have also included a `test.wav` file. Just something I threw together in GarageBand. The notebook works pretty well with it! You can comment this line: `fn_wav = os.path.join('..', 'data', 'C5', 'FMP_C5_F01_Beatles_LetItBe-mm1-4_Original.wav')` and uncomment this line:  `fn_wav = os.path.join('..', 'test.wav')` to see the results. Note that the colours will be wrong as it is still displaying chords from the Beatles song.
