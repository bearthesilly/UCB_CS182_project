# UCB_CS182_project
## Logistics

[IMPORTANT!] We have deposited our code in different branches by authors in this repo. We will organize them up and put them in the `main` branch with detailed instruction for reproducing by the EOD. Thank you for your patience! (13:00 12/8)

[RELEASED!] **We have organized the code for reproduction** and released the instructions to easily reproduce the results in the following `How to reproduce section`! Thank you again for your patience. (16:30 12/8)

## How to reproduce

As you've noticed in the paper and in this repo: We have many sub-experiments! Here are the simple steps to reproduce them!

1. Instantiate a conda environment, and run the following bash line to install necessary dependencies:

````bash
pip install -r requirement.txt
````

2. For the sub-experiment you want to reproduce, **enter into its corresponding folder** as the work directory, and follow the instructions in that folder's `README.md`

All our experiments are run on a single A100 GPU.

## ***Explorations on Catastrophic Forgetting Mitigation Curriculum***

### Abstract
Catastrophic forgetting (CF), the phenomenon that occurs when a model forgets previously learned data as a result of learning new data, poses a significant barrier to lifelong learning of LLMs, where many factors of fine-tuning tasks within different domains can ultimately degrade the performance on previously learned tasks. In this project report, we propose different curriculum designs to mitigate catastrophic forgetting and conduct experiments to verify the effectiveness of these approaches. Subsequently, we investigate whether curriculum design exhibits a scalable transferability effect by using a smaller LLM to approximate the ideal curriculum for a larger LLM from the same family to reduce training power consumption and increase downstream performance. 

### Method

![image](figure/figure.png)

### Poster

![image](figure/poster.png)

### Conclusion

So far, we have successfully reproduced the catastrophic forgetting phenomenon and verified our proposed strategies. Through the reproduction of CF, we gain some intuition about this phenomenon and accordingly have some thoughts and ideas on mitigation strategies. We elaborate on the motivation and implementation of these strategies. Some of our strategies worked, whereas some of them did not. We also conduct a preliminary investigation into curriculum transfer for LLM continual learning through a case study.

### Limitation

First, the scope of our experiments is not exhaustive. Due to time and resource constraints, we could not achieve comprehensive coverage in terms of strategy validation or comparative baseline design. Second, given the timeline and the large volume of experiments, we were unable to conduct repeated trials with multiple random seeds, precluding the inclusion of error bars for a more rigorous statistical assessment. Finally, while the methodological novelty of this work may be limited as a preliminary exploration into catastrophic forgetting, it has provided us with deep insights into the field through extensive literature review, strategy formulation, and empirical validation. 

Note that for the experiment of curriculum transfer, it should be noted that the sample size settings used in this study were not optimally tuned, which resulted in accelerated convergence rates and artificially smoothed trajectories for the curriculum strategy. While we recognize this as an experimental oversight, strict time constraints precluded us from re-conducting the experiments to rectify this issue.

### Acknowledgement

This is a joint research work by:

````text
Aaron Zheng: aaronz@berkeley.edu
Zhangzhi Xiong: xiongzhzh2023@berkeley.edu
Jason Lee: jason_lee@berkeley.edu
Andy Zhang: zhangnd16@berkeley.edu
Tianyu Gu: gty2005@berkeley.edu
````

**Feel free to email us if you encounter any problem!** **Also, we'd like to express our sincere gratitude to peer reviewers and professors! It is your advice that keep us improving!** 

