
**Generizable, Heterogeneous Multi-Agent Traffic Simulator (MATS)**
===========================================================



[![DOI](https://zenodo.org/badge/805019355.svg)](https://doi.org/10.5281/zenodo.13884942)



The Generizable Heterogeneous Multi-Agent Traffic Simulator (MATS) is a novel tool to gamify real traffic by utilizing a digital-twin of real traffic scenes. The proposed model here leverages both Supervised and Reinforcement Learning techniques to model complex traffic scenarios with scalabilty and realism.

**Key Features:**

* **Generizability**: Robustly tested on two distinct environments:
	+ Intersections
	+ Shared spaces
* **Heterogeneity**: Simulates the behavior of multiple types of agents, namely:
	+ Vehicles
	+ Pedestrians
	+ Cyclists
* **Multi-Agent**: Enables simultaneous stepping for all agents with a short prediction horizon of 400 ms using parameter sharing
* **Realism**: Trained on real traffic trajectories to imitate it in both short-term (with supervised learning) and long-term (with reinforcement learning)


<p align="center">
  <img src="methodology.svg" style="background-color:white;"/>
</p>

The provided codebase includes comprehensive scripts for both model training and simulation. The latter can be executed independently to test various multi-agent traffic models, empowering new works to explore and validate different Multi-agent Reinforcement Learning (MARL) algorthems.


## Videos of Simulation Results


The following are video demonstrations of MATS, for increasing number of agents (up to 96) in both environement. Counters for collisions and moving outside the road are shown.

<!--![](ind_model/InD.gif)-->


<table>
  <thead>
    <tr>
      <th><center>Intersection Case</center></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <img src='ind_model/InD.gif' width="100%" />
      </td>
    </tr>
    <tr>
      <th><center>Shared Space Case</center></th>
    </tr>
    <tr>
      <td>
        <img src='unid_model/UniD.gif' width="100%" />
      </td>
    </tr>
    <tr>
  </tbody>
</table>




## Getting Started:

### Step 1
Clone the repo

### Step 2
Intsall dependices with: 

`pip install -r requirements.txt`

### Step 3

**Run the Trained Model**

To run the trained model and observe its behavior, navigate to either `unid_model` or `ind_model` and execute:

```bash
python trafficenv_D.py
```

### Step 4 (optional)

Additionally, to load and retrain the supervised learning model, it is possible to rerun `bc.py`, but the preprocessed numpy files are needed first.

These files can be generated by downloading the datasets of IND and UNID in `ind_model/indds` and `unid_model/unids` respectively. Then going through the code in the notebooks `ind_model/ind_preprocessing.ipynb` and `unid_model/unid_preprocessing.ipynb`. 


To view the prediction results and the preprocessed data, check the notebooks  `ind_model/draw_results.ipynb` and `unid_model/draw_results.ipynb`


## Citation:



```bibtex
@inproceedings{yousif2024integrating,
  title={Integrating Supervised and Reinforcement Learning for Heterogeneous Traffic Simulation},
  author={Yousif, Yasin M and M{\"u}ller, J{\"o}rg P},
  booktitle={International Conference on Practical Applications of Agents and Multi-Agent Systems},
  pages={277--288},
  year={2024},
  organization={Springer}
}

```









## Help

If you need help, please create a new issue by clicking the "New issue" button on our repository's issue tracker page. Provide as much detail as possible about the issue you're experiencing, including:

* A clear description of the problem
* Any relevant code snippets or error messages
* The steps you've taken to troubleshoot the issue

