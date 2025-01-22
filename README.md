# Alpha

A simple project to implement the AlphaZero algorithm on the popular board game [Blokus](https://www.officialgamerules.org/blokus).

This project was inspired by the work of deepmind in the creation of the AlphaGo,
AlphaGo Zero, Alpha Zero and MuZero algorithms (respetively below): 

- [Lessons learnt from AlphaGo](https://bit.ly/2uCqK2S)

- [AlphaGoZero](https://go.nature.com/385X3F3)

- [AlphaZero](https://bit.ly/2wYuIns)

- [MuZero](https://bit.ly/2uLwl7i)

- [MuZero Neurips poster 2019](https://bit.ly/2VwRJbf)

- [MuZero pseudocode](https://arxiv.org/src/1911.08265v1/anc/pseudocode.py)

Additionally, this codebase is heavily inspired by:
- [Alpha Zero General](https://github.com/suragnair/alpha-zero-general)

## Project Outline

I have discretised the steps of the project into manageable stages of (hopefully)
achievable goals as follows:

### Tic-Tac-Toe

Training an algorithm to play this simple game is a quick check the basics are working.

- [x] Can you get a framework to work for the simple game of Tic-Tac-Toe
- [x] Enable logging for debugging in future runs
- [ ] Enable time logging 
- [ ] Experiment with ResNet architecture
- [ ] Add a learning rate scheduler
- [ ] Figure out how to buffer training examples 
- [ ] Draw with/be competitive with a perfect Tic-Tac-Toe bot

### Connect 4

- [ ] Can I train the algorithm faster than other sources i.e in this [article](https://medium.com/oracledevs/lessons-from-alpha-zero-part-6-hyperparameter-tuning-b1cfcbe4ca9a)
- [ ] Experiment with parameter tuning to optimise later work

### Blokus

- [ ] Design input and figure out representation for use with net
- [ ] Figure out board representation and efficient algorithm for calculating available moves (using caching)
- [ ] Optimise training time

### Become the best blokus playing AI on the planet
Play 100 games of blokus against the best Blokus AIs in the world and win > 50%
of the games.

- [Pentobi AI](https://pentobi.sourceforge.io/)

- [FGPA AI](https://bit.ly/2TtjRcv) 

