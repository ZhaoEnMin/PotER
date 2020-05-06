# Introduction
This repository is an example of [PotER 2020 Potentialized Experience Replay]in Tensorflow.
```
@inproceedings{Zhao2020PotER,
  title={Potential Driven Reinforcement Learning\\for Hard Exploration Tasks},
  author={Enmin Zhao and Shihong Deng and Yifan Zang and Yongxin Kang and Kai Li and Junliang Xing},
  booktitle={IJCAI},
  year={2020}
}
```
Our code is based on [OpenAI Baselines](https://github.com/openai/baselines) and [SIL](https://github.com/junhyukoh/self-imitation-learning).

# Training
The following command runs `PotER+SIL` in first room of Montezuma's Revenge:
```
CUDA_VISIBLE_DEVICES=0 python3 run_atari_PotERsil.py --env MontezumaRevengeNoFrameskip-v4
```



