# Downloading the EgoProceL dataset

The EgoProceL dataset was released in an ECCV 2022 publication titled: "[My View is the Best View: Procedure Learning from Egocentric Videos](https://sid2697.github.io/egoprocel/)".

Link to download EgoProceL: [https://sid2697.github.io/egoprocel/#download](https://sid2697.github.io/egoprocel/#download)

Link to the project page: [https://sid2697.github.io/egoprocel/](https://sid2697.github.io/egoprocel/)

Link to the paper: Coming soon!

This document summarizes the steps required to download the videos and annotations in EgoProceL.

## Downloading the annotations

Link to download: ~[OneDrive](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/siddhant_bansal_research_iiit_ac_in/EgqvXb5syepDv1z-UAwsYEQBivEYauz8tuotty7eey32Ng?e=TNXpBE)~ (expired) [Google Drive](https://drive.google.com/drive/folders/17u7ReqPOJ29lbVZT8PDEd5dN17VBqS5Q?usp=share_link) (please use this)

Folder structure:

```
+ annotations
    + CMU-MMAC
        + Brownie
            + S07_Brownie_Video
                - S07_Brownie_6510211-1103.csv
                - S07_Brownie_7150991-1431.csv
                - S07_Brownie_7151020-1103.csv
                - S07_Brownie_7151062-1103.csv
                - S07_Brownie_8421130-2374.csv
            ...
        + Eggs
            + S07_Eggs_Video
                - S07_Eggs_6510211-1110.csv
                ...
            ...
        ...
    + EGTEA_Gaze+
        + BaconAndEggs
            - OP01-R03-BaconAndEggs.csv
            - OP02-R03-BaconAndEggs.csv
            ...
        + Cheeseburger
            ...
        ...
    + EPIC-Tents
        - 01.tent.090617.gopro.egoprocel.ann.csv
        - 02.tent.120617.gopro.egoprocel.ann.csv
        ...
    + MECCANO
        - 0003.csv
        - 0004.csv
        ...
    + pc_assembly
        - Head_5.csv
        - Head_7.csv
        ...
    + pc_disassembly
        - Head_6.csv
        - Head_8.csv
        ...
```

Things to note about the annotations:
1. The annotation file name exactly matches the video's file name.
1. The annotation `csv` contains three columns, a) key-step's start second, b) key-step's end second, c) name of the key-step.
1. The datasets with multiple categories (e.g., CMU-MMAC) have multiple directories under them. In contrast, datasets with single category (e.g., PC Assembly) directly have the annotation `csv`.

## Downloading the videos

The videos in EgoProceL were obtained from multiple sources. Here we list the steps to download the videos from each of the sources:

### PC Assembly and Disassembly
These videos were recorded by [Pravin Nagar](https://scholar.google.com/citations?user=k4TZSPQAAAAJ&hl=en) and [Sagar Verma](https://sagarverma.github.io/) at IIIT Delhi.

Download link: ~[OneDrive](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/siddhant_bansal_research_iiit_ac_in/Ev14SL5JYtJNpVUUAhDMgEABbPnTYpbDUzBYAhQToyHmVw?e=cQu5by)~ [Google Drive](https://drive.google.com/drive/folders/19QofEX1tcfmU3fO2DAwGV8iiBZPAiRuI?usp=share_link) (please use this)

### CMU-MMAC
CMU-MMAC videos can be downloaded from [http://kitchen.cs.cmu.edu/main.php](http://kitchen.cs.cmu.edu/main.php).

Here is a script to download all the videos at once: [https://github.com/Sid2697/EgoProceL-egocentric-procedure-learning/blob/main/misc/CMU_Kitchens/download.py](https://github.com/Sid2697/EgoProceL-egocentric-procedure-learning/blob/main/misc/CMU_Kitchens/download.py).

### EGTEA-Gaze+
EGTEA-Gaze+ videos can be downloaded from [https://cbs.ic.gatech.edu/fpv/](https://cbs.ic.gatech.edu/fpv/).

### EPIC-Tents
EPIC-Tents videos can be downloaded from [https://sites.google.com/view/epic-tent](https://sites.google.com/view/epic-tent).

### MECCANO
MECCANO videos can be downloaded from [https://iplab.dmi.unict.it/MECCANO/](https://iplab.dmi.unict.it/MECCANO/).

Things to note about the videos:
1. It is recommended to save the videos following the annotation directory's structure.
1. Due to compatibility reasons (mentioned in the paper), not all the videos from each dataset have been used for the task. Please refer to the available annotation files to get an idea of which videos to use.

### Contact

In case of any concern contact [Siddhant Bansal](https://sid2697.github.io).
Email: [siddhant.bansal@research.iiit.ac.in](mailto:siddhant.bansal@research.iiit.ac.in)

Please consider citing if you make use of the EgoProceL dataset and/or the corresponding code:

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
