# Character Level LSTM - Tensorflow/MLflow

This repository contains an Mlflow deployment of my character-level LSTM. Since the LSTM executes custom logic when evaluating queries, it is represented as an `mlflow.pyfunc` model.
<p align="center">
<img src="https://i.imgur.com/5tCB71T.png" width="50%"/>
</p>


### Installation
1. Clone repo locally
2. Install dependencies with `pip3 install -r requirements.txt`
3. Train model from scratch or download saves from the link below into the saves folder
3. Run `python3 create_model_wrapper.py` to create pyfunc model in `/model_path`
4. Run `python3 application.py` to start the flask server

### Architecture
I trained a standard multi-layer cahracter level LSTM with the following cell archictecture. Architecture was heavily inspired by Andrej Karpathy's blog post http://karpathy.github.io/2015/05/21/rnn-effectiveness/
![LSTM](https://i.imgur.com/gxBbaX2.png)

The network has three stacked layers of cells with a batch sequence length size of 100. I trained on an single NVIDIA GeForce 1080.

Download pre-trained model here: https://www.dropbox.com/s/o0glxwlbwtesdck/saves.zip?dl=0

Random weights:

>U—CN3gLBddDWÁu!HÁ”;ap/8!ôdPEç’hÁóblOxD/ç)Vsn@i)Á”YáPxwDöAhDœ “uwDóJ$—u)cDèBDü->èAàZ@zàB(ióH$äaFu3ýYmFniRLMÉKmYwBwZO@Uuaîq@AG()%aèLî$5b%üpïER=—‘À*mê?ç:V
>v
>B’BlcFIy5hf98JouV﻿ jAúoöt(y(WWnRE;él:xGlêEWÀnoG

Here are some examples of text halucinations:

>the ball, calmls happened to you, does he cavered in the steps to maken, narged the wortured loud. The doctor and from his ??
>fatare and ran now a run
>into the existence of the great some! They were saying, he felt a drunk as
>you weight he saw in its away you on leaving a reflection”arch, and all they had till the army and are no waiting those 
>people was heart free terringly with
>somewhere.
>
>“To pass? Dudly—retelies to them said for his place of Austria consulting
>impossible, more still told his corrigencial it had way out of honors
>could not been glanced lea into the comrades
>merely to a relate.
>
>Without will be a the people she told not your house, looking began unterestby plump ask.
>
>The old fair that that it he has escort of our dresses ready, but his dwant to you wagrag immediately if bridge and rouding 
>to kiss a whit of relay of, he must do so, is not imaginex regard
>to really singers,
>and exembrowed smiles, stapid vicious own ind-Ardisoners, quition, and I have letter she
>rose and


![Training](https://i.imgur.com/2KG7cO5.png)
