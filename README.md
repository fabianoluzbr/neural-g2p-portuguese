# Neural G2P to portuguese language

Grapheme-to-phoneme (G2P) conversion is the process of generating pronunciation for words based on their written form. It has a highly essential role for natural language processing, text-to-speech synthesis and automatic speech recognition systems. This project was adapted from https://github.com/hajix/G2P.

## Dependencies
The following libraries are used:<br/>
pytorch<br/>
tqdm<br/>
matplotlib<br/>

Install dependencies using pip:
```
pip3 install -r requirements.txt
```

## Dataset
The dataset used here was taken from site http://www.portaldalinguaportuguesa.org/, as well as some insertions made by me so that the dataset would give more coverage to common words in the daily life of the Brazilian Portuguese. Some ambiguities were also resolved as the intent of this dataset is to contain a specific speaker bias. The dictionary based on S?o Paulo speakers was chosen.

As in https://github.com/hajix/G2P, on which this implementation was based, you could easily provide and use your own language specific pronunciatin doctionary for training G2P.
More details about data preparation and contribution could be found in ```resources```.<br/>
Feel free to provide resources for other languages.

## Attention Model
Both encoder-decoder seq2seq model and attention model could handle G2P problem.
Here we train attention based model.
![attention model](attention/attention-bidi.jpg)
The encoder model get sequence of graphemes and produces states at each timestep.
Encoder states used during attention decoding.
The decoder attends to appropriate encoder state (according to its state) and produces phonemes.


### Train
To start training the model run:
```
python train.py
```
You can also use tensorboard to check the training loss:
```
tensorboard --logdir log --bind_all
```
Training parameters could be found at ```config.py```.

### Test
To get pronunciation of a word:
```
# PT-BR example
python inference.py --sentence 'olá, vamos testar esse projeto.'
o|l|a| |,| |v|a|m|ʊ|s| |t|e|s|t|a| |e|s|i| |p|ɾ|o|ʒ|e|t|ʊ| |.

```
You could also visualize the attention weights, using ```--visualize```:
```
# PT-BR example
python inference.py --visualize --word 'olá, vamos testar esse projeto.'
o|l|a| |,| |v|a|m|ʊ|s| |t|e|s|t|a| |e|s|i| |p|ɾ|o|ʒ|e|t|ʊ| |.
```
