# Neurosky_SpaceInvader
## About
Project of COGS 189 class in UCSD which uses neurosky mindwave headset to play space invader game in python. In this project, we made the space invader game with which we can utilize human brain signals while playing game. You will get an advantage while you focus on game (attention is high) and get a disadvantage when you lose your calm (meditation is low).  
Original code of space invader from: https://github.com/leerob/Space_Invaders.  
  
<img src="https://github.com/hiroshi2727/Neurosky_SpaceInvader/blob/master/images/playing_game.png" height="50%" width="50%"/>

## Instruction
First, you solve few math questions to decide attention and meditation threshold values.  
Then, you choose the level of game between 1 and 6, and you can start to play game.
While playing game, use arrow keys to move charater and space to shoot bullet(s).  
  
## Requirements
Windows PC  
Python 2.7.15  
Pygame 1.9.4  
NeuroPy 0.1  
Neurosky Mindwave Headset  

## Set Up  
1) Create a conda Python 2.7.15 environment  
2) Download and extract the NeuroPy.zip from: https://github.com/lihas/NeuroPy/tree/master/dist  
3) Extract the zip, and navigate to the directory with the command prompt
4) With your activated Python 2.7.15, run python setup.py install
5) After NeuroPy is installed install pyserial via:
```
pip install pyserial
```
6) Change the variable "PORT 1" in line 17 of spaceinvader.py to your COM or dev location  
7) Run the following code:
```
python spaceinvaders.py
```

# NeuroSky
