### A Dual System Framework for Dynamic Routine travel Behavioral Modeling

This repository contains the replication data and code for \
\
`Luo & Walker (n.d.) Incorporating the Notion of Habit from Neuroscience in Dynamic Routine Travel Modeling: an Application to Teleommute Decisions through the Pandemic`

The paper is submitted to [Transportation Research Part B: Methodological](https://www.sciencedirect.com/journal/transportation-research-part-b-methodological)

### Code
Model estimation results in the paper can be replicated by running `multiprocessing.py` and `Multiprocessing_bootstrap.py` notebook.
- `Trainer.py` provides the training file
- `agents.py` provides the
- `optim.py` provides
- `utils.py` provides

### Data
- The data used in this work is from GPS data collected by Embee Mobile. The dataset we provide (`demo.csv`) is the post-processed data sufficient to replicate the estimation results in the paper.

- Commute choice trajectories for 573 full-time workers are included:

- For each trajectory (`_trajectory.csv`)

- Demographic information for each workers is included in `demo.csv`, including household income, age, gender, education level, distance from home to work, and occupation status (`full-time` or not).
  - `panelist_id`
  - `hh_income`
  - `age`
  - `gender`
  - `edu_level`
  - `dfwh`
  - `occupation_status`


### Environment
The code is written in python 3. Packages and library required include:
- `numpy`
- `pandas`
- `datetime`
- `os`
- `glob`
- `re`
