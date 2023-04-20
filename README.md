<h3 align="center">Robotic Palletizing Problem using RL</h3>

## 📝 Table of Contents

- [About](#about)
- [Built Using](#built_using)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [TODO](#todo)
- [Authors](#authors)

## 🧐 About <a name = "about"></a>

Modelled the palletizing problem as a POMDP and solved it using a Soft-Actor Critic Method

## Considerations

- The boxes will be coming in an online fashion and a particular sequence
- Boxes can be rotated before placement
- Placement done in top-down approach
- Box not placed successfully is discarded
- All 6 orientations of a box are considered stable

## Theory

### State

- State is made of 2 components. $s_p, s_b$

1. The palette state $s_p$ can be understood as the height state of the pallete. It's of size $L\times W$
2. The box state $s_b$ captures the state of the boxes. $s_b = \{b_i\} i=\{1,...,n}$ where $n$ is the number of boxes in the sequence. $b_i$ = $l_i, w_i, h_i, P_i$. Where $P_i = 0$ if the box is yet to be loaded, $1$ if is loaded successfully, $-1$ if discarded.

### Action

Since the boxes are placed in a sequence, we only care of the boxes orientation and location placed. Hence $a = (a_o, a_l)$, $a_o_i \in \{(l_i, w_i, h_i), (l_i, h_i, w_i), (w_i, l_i, h_i), (w_i, h_i, l_i), (h_i, l_i, w_i), (h_i, w_i, l_i)} and $a_l is a coordinate of where the front, lower, left corner of the box is placed in the palette.

### Observation

At each timestep, only a certain segment of the sequence is visible to the agent

### Reward

We wish to maximise for maximum volume packed in the pallete. Hence the reward is the fraction of the box packed of the total palette volume. The reward is 0 if the box is discarded. Additionally, there is an instability penalty, which is measured using local differencing method: Taking the RMSE of the convolution with filter of [[-1, -1, -1],[-1, 8, -1], [-1, -1, -1]] and normalizing with $L \times W$

### Agent

I used a Soft Actor Critic network with a double Q network for the critic.

## 🏁 Getting Started <a name = "getting_started"></a>

### Installing

Create the environment

`conda env create -f environment.yml`

## 🎈 Usage <a name="usage"></a>

Activate environment `conda activate PalletizerEnv`

Run `python main.py`

## Future Steps <a name="todo"></a>

- Consider using a PCT for POMDP structure
- Incorporate a transformer based SAC network

## ✍️ Authors <a name = "authors"></a>

- [@saksham36](https://github.com/saksham36)
