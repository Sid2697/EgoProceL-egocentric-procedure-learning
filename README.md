# My View is the Best View: Procedure Learning from Egocentric Videos

[![arXiv](https://img.shields.io/badge/cs.cv-arXiv%3A2207.10883-42ba94.svg)](http://arxiv.org/abs/2207.10883)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

### [Project page](https://sid2697.github.io/egoprocel/) | [**Download EgoProceL**](https://sid2697.github.io/egoprocel/#download) | [Paper](https://arxiv.org/pdf/2207.10883) | [Video](https://sid2697.github.io/docs/1886.mp4) | [Poster](https://sid2697.github.io/docs/1886.pdf) | [Springer](https://link.springer.com/chapter/10.1007/978-3-031-19778-9_38)

This repository contains code for the paper

"**My View is the Best View: Procedure Learning from Egocentric Videos**" *[Siddhant Bansal](https://sid2697.github.io), [Chetan Arora](https://www.cse.iitd.ac.in/~chetan/), [C.V. Jawahar](https://faculty.iiit.ac.in/~jawahar/index.html)* 
published in [ECCV 2022](https://eccv2022.ecva.net/).

## Abstract
Procedure learning involves identifying the key-steps and determining their logical order to perform a task. Existing approaches commonly use third-person videos for learning the procedure. This makes the manipulated object small in appearance and often occluded by the actor, leading to significant errors. In contrast, we observe that videos obtained from first-person (egocentric) wearable cameras provide an unobstructed and clear view of the action. However, procedure learning from egocentric videos is challenging because (a) the camera view undergoes extreme changes due to the wearer's head motion, and (b) the presence of unrelated frames due to the unconstrained nature of the videos. Due to this, current state-of-the-art methods' assumptions that the actions occur at approximately the same time and are of the same duration, do not hold. Instead, we propose to use the signal provided by the temporal correspondences between key-steps across videos. To this end, we present a novel self-supervised Correspond and Cut (CnC) framework for procedure learning. CnC identifies and utilizes the temporal correspondences between the key-steps across multiple videos to learn the procedure. Our experiments show that CnC outperforms the state-of-the-art on the benchmark ProceL and CrossTask datasets by 5.2% and 6.3%, respectively. Furthermore, for procedure learning using egocentric videos, we propose the EgoProceL dataset consisting of 62 hours of videos captured by 130 subjects performing 16 tasks.

## Download EgoProceL

Downloading instructions: [https://github.com/Sid2697/EgoProceL-egocentric-procedure-learning/blob/main/EgoProceL-download-README.md](https://github.com/Sid2697/EgoProceL-egocentric-procedure-learning/blob/main/EgoProceL-download-README.md)

### Overview of EgoProceL

<video class="centered" width="100%" autoplay muted loop playsinline>
  <source src="https://sid2697.github.io/images/projectpic/EgoProceL-demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

EgoProceL consists of
- <b><u>62</u> hours</b> of videos captured by
- <b><u>130</u> subjects</b>
- performing <b><u>16</u> tasks</b>
- maximum of <b><u>17</u> key-steps</b>
- average <b><u>0.38</u> foreground ratio</b>
- average <b><u>0.12</u> missing steps ratio</b>
- average <b><u>0.49</u> repeated steps ratio</b>

A portion of EgoProceL consist of videos from the following datasets:
- [CMU-MMAC](http://kitchen.cs.cmu.edu/main.php)
- [EGTEA Gaze+](https://cbs.ic.gatech.edu/fpv/)
- [MECCANO](https://iplab.dmi.unict.it/MECCANO/)
- [EPIC-Tent](https://sites.google.com/view/epic-tent)


## Training the Embedder Network in the Correspond and Cut Framework

![CnC](https://sid2697.github.io/images/projectpic/ECCV_diagrams-Methodology_v0-5.png)
CnC takes in multiple videos from the same task and passes them through the embedder network trained using the proposed TC3I loss. The goal of the embedder network is to learn similar embeddings for corresponding key-steps from multiple videos and for temporally close frames.

To train the embedder network, use the following instructions:

1. Clone the repository and enter it

```bash
git clone https://github.com/Sid2697/EgoProceL-egocentric-procedure-learning
cd EgoProceL-egocentric-procedure-learning
```

2. Create a virtual environment (named `egoprocel`) and install the required dependencies. A handy guide for `miniconda` ([link](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)); installing `miniconda` ([link](https://docs.conda.io/en/latest/miniconda.html)).

```python
conda create --name egoprocel python=3.9
pip install -r requirements.txt
```

Before moving to the next step, we strongly recommend to have a look at `config.py` ([https://github.com/Sid2697/EgoProceL-egocentric-procedure-learning/blob/main/configs/config.py](https://github.com/Sid2697/EgoProceL-egocentric-procedure-learning/blob/main/configs/config.py)). It contains a detailed documentation of all the variables used in this repository.

3. Refer to `demo_config.yaml` ([https://github.com/Sid2697/EgoProceL-egocentric-procedure-learning/blob/main/configs/demo_config.yaml](https://github.com/Sid2697/EgoProceL-egocentric-procedure-learning/blob/main/configs/demo_config.yaml)). This config file will allow us to train the Embedder network on the sandwich category in CMU-MMAC subset of EgoProceL. Update the paths accordingly. For example, update path to videos (`CMU_KITCHENS.VIDEOS_PATH`), annotations (`CMU_KITCHENS.ANNS_PATH`), directory to extract frames (`CMU_KITCHENS.FRAMES_PATH`), what task to train (`ANNOTATION.CATEGORY`), etc.

4. Once the paths are updated and correct, run the following command to train EmbedNet:

```python
python -m RepLearn.TCC.main --cfg configs/demo_config.yaml
```

The trained models will be saved in `LOG.DIR`.

## Testing the Trained Network (generating the embeddings and clustering the frames)

Once the network is trained, it can be tested using the following command:

```python
python -m RepLearn.TCC.procedure_learning --cfg configs/demo_config.yaml
```

Remember to point to the path of the saved model using `TCC.MODEL_PATH`. This can be done from the command line too:

```python
python -m RepLearn.TCC.procedure_learning --cfg configs/demo_config.yaml TCC.MODEL_PATH path/to/the/model.pth
```

## Citation

Please consider citing the following work if you make use of this repository:

```
@InProceedings{EgoProceLECCV2022,
author="Bansal, Siddhant
and Arora, Chetan
and Jawahar, C.V.",
title="My View is the Best View: Procedure Learning from Egocentric Videos",
booktitle = "European Conference on Computer Vision (ECCV)",
year="2022"
}
```


Please consider citing the following works if you make use of the EgoProceL dataset:

```
@InProceedings{EgoProceLECCV2022,
author="Bansal, Siddhant
and Arora, Chetan
and Jawahar, C.V.",
title="My View is the Best View: Procedure Learning from Egocentric Videos",
booktitle = "European Conference on Computer Vision (ECCV)",
year="2022"
}

@InProceedings{CMU_Kitchens,
author = "De La Torre, F. and Hodgins, J. and Bargteil, A. and Martin, X. and Macey, J. and Collado, A. and Beltran, P.",
title = "Guide to the Carnegie Mellon University Multimodal Activity (CMU-MMAC) database.",
booktitle = "Robotics Institute",
year = "2008"
}

@InProceedings{egtea_gaze_p,
author = "Li, Yin and Liu, Miao and Rehg, James M.",
title =  "In the Eye of Beholder: Joint Learning of Gaze and Actions in First Person Video",
booktitle = "European Conference on Computer Vision (ECCV)",
year = "2018"
}

@InProceedings{meccano,
    author    = "Ragusa, Francesco and Furnari, Antonino and Livatino, Salvatore and Farinella, Giovanni Maria",
    title     = "The MECCANO Dataset: Understanding Human-Object Interactions From Egocentric Videos in an Industrial-Like Domain",
    booktitle = "Winter Conference on Applications of Computer Vision (WACV)",
    year      = "2021"
}

@InProceedings{tent,
author = "Jang, Youngkyoon and Sullivan, Brian and Ludwig, Casimir and Gilchrist, Iain and Damen, Dima and Mayol-Cuevas, Walterio",
title = "EPIC-Tent: An Egocentric Video Dataset for Camping Tent Assembly",
booktitle = "International Conference on Computer Vision (ICCV) Workshops",
year = "2019"
}
```

## Acknowledgements

Code in this repository is build upon or contains portions of code from the following repositories. We thank and acknowledge authors for releasing the code!

- The evaluation script and baseline code was referred from [https://github.com/hbdat/eccv20_Multi_Task_Procedure_Learning](https://github.com/hbdat/eccv20_Multi_Task_Procedure_Learning)
- The code for TCC was referred from [https://github.com/yukimasano/self-label](https://github.com/yukimasano/self-label) and [https://github.com/Fujiki-Nakamura/TCCL.pytorch](https://github.com/Fujiki-Nakamura/TCCL.pytorch).
- Jupyter Notebooks for TCC [https://github.com/google-research/google-research/tree/master/tcc](https://github.com/google-research/google-research/tree/master/tcc)


## Contact

In case of any issue, feel free to create a pull request. Or reach out to [Siddhant Bansal](https://sid2697.github.io).
