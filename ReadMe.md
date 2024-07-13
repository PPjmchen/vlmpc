# VLMPC: Vision-Language Model Predictive Control for Robotic Manipulation

This is the official repo for our Paper: [VLMPC: Vision-Language Model Predictive Control for Robotic Manipulation](https://roboticsconference.org/program/papers/106/), which is accepted by **RSS2024**. 

We provide the implementation of VLMPC in [Language-Table](https://github.com/google-research/language-table) environment.

![framework image](./framework.jpg)



## Installation

- We recoomend using conda environment:

```bash
conda create -n vlmpc python=3.10
conda activate vlmpc
```

- Install dependencies:

```bash
pip install -r requirements.txt
```

- Add your [Openai API](https://openai.com/index/openai-api/) key to the vlmpc.py.


## Quickstart

Example command to run the experiments.
```python
python main.py --checkpoint_file path/to/video_prediction_model/checkpoint --task push_corner --zoom 0.03 --num_samples 50 --plan_freq 3 --det_path path/to/yolo/checkpoint
```

## Code Structure

Core to VLMPC:
- ```main.py``` 
- ```vlmpc.py```: Implementation of the MPC loop for combining VLMs to guide the robotic manipulation step by step.
- ```prompt_gpt.py```: Prompts used to decompose task and generate the sub-goals.
- ```video_interface.py```: Interface that provide video prediction model to predicture future states.
- ```sampler.py```: Interface used for sampling actions.

## Acknowledgements
- Environment is based on [Language-Table](https://interactive-language.github.io/).
- The implementation of DMVFN-Act video prediction model is based on [DMVFN](https://github.com/hzwer/CVPR2023-DMVFN)
- pysort


## Citation

If you find this work useful, please feel free to cite our paper or leave a star:
```
```