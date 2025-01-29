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
- [AplhaZero](https://github.com/michaelnny/alpha_zero?tab=readme-ov-file)

## Project Outline

I have discretised the steps of the project into manageable stages of (hopefully)
achievable goals as follows:

### Tic-Tac-Toe

Training an algorithm to play this simple game is a quick check the basics are working.

- [x] Can you get a framework to work for the simple game of Tic-Tac-Toe
- [x] Enable logging for debugging in future runs
- [x] Enable time logging 
- [x] Figure out how to buffer training examples
- [x] Experiment with ResNet architecture
- [x] Draw with/be competitive with a perfect Tic-Tac-Toe bot

### Blokus Duo

- [x] Design input and figure out representation for use with net
- [ ] Figure out board representation and efficient algorithm for calculating available moves (using caching)
- [ ] Optimise training time
- [ ] Add a learning rate scheduler

Can use connect 4 as a debugging step if the above isn't working

### Become the best blokus playing AI on the planet
Play 100 games of blokus against the best Blokus AIs in the world and win > 50%
of the games.

- [Pentobi AI](https://pentobi.sourceforge.io/)

- [FGPA AI](https://bit.ly/2TtjRcv) 

