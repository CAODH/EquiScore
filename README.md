# EquiScore is a generic protein-ligand interaction scoring method integrating physical prior knowledge with data augmentation modeling

### [Paper on BioRXiv](https://www.biorxiv.org/content/10.1101/2023.06.18.545464v2)

Implementation of EquiScore, by Duanhua Cao 😊.

### 🔔 News

   Some bugs(🐛 🐛) have been fixed, and bash commands are further provided to help users unfamiliar with python quickly use EquiScore for virtual screening 😃

**This repository contains all code, instructions and model weights necessary to **screen compounds** by EquiScore, eval EquiScore or to retrain a new model.**

If you have any question, feel free to open an issue or reach out to us: [caodh@zju.edu.cn ✉️](caodh@zju.edu.cn).

## Framework

![Alt Text](./figs/model_framework.png)

## Dataset

If you want to train one of our models with the PDBscreen data you should do:

1. download Preprocessed PDBscreen data from [zenodo](https://doi.org/10.5281/zenodo.8049380)
2. uncompress the directory by tar command and place it into `data` such that you have the path `/EquiScore/data/training_data/PDBscreen`
3. see retraining EquiScore part for details.

## Setup Environment

We recommend setting up the environment using [Anaconda](https://docs.anaconda.com/anaconda/install/index.html).

   Clone the current repo

   git clone git@github.com:CAODH/EquiScore.git

This is an example for how to set up a working conda environment to run the code (but make sure to use the correct pytorch, DGL, cuda versions or cpu only versions):

   `conda create --name EquiScore python=3.8`
   `conda activate EquiScore`

   **Through our testing, the relevant environment can be successfully installed by executing the following commands in sequence:**

   `conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch`

   `conda install -c dglteam dgl-cuda11.1`

   `conda install -c conda-forge rdkit`

   `conda install -c conda-forge biopython`

   `conda install -c conda-forge scikit-learn`

   `conda install -c conda-forge prolif`

   `pip install prefetch-generator`

   `pip install lmdb`

   `pip install numpy==1.22.3`

**or you can download the conda-packed file [zenodo](https://zenodo.org/records/10417452), and then unzip it in ${anaconda install dir}/anaconda3/envs/EquiScore. ${anaconda install dir} represents the dir where the anaconda is installed. For me, ${anaconda install dir}=/root.**

   `mkdir ${anaconda install dir}/anaconda3/envs/EquiScore `


   `tar -xzvf EquiScore.tar.gz -C ${anaconda install dir}/anaconda3/envs/EquiScore`


   `conda activate EquiScore`


   **after enter the EquiScore env: run**


   `conda-unpack`

## **Using the provided model weights to screen a compound for your target protein**

   We implemented a Screening.py python script, to help **anyone want to screen Hits from a compound library**.

   We provide a toy example under the ./data/sample_data folder for illustration.

### **🚀just some steps needed**

1. Docking compounds with target protein to get docking pose, EquiScore is robust to pose sources and you can choose any method you are familiar with to generate poses(**Glide,Vina,Surflex,Gold,LeDock**), or you can try a **deep learning method**.
2. Assume that you have obtained the results of the docking in the previous step. Then, get pocket region and compound pose.
   run script:

   `python ./get_pocket/get_pocket.py --docking_result ./data/sample_data/sample_compounds.sdf --recptor_pdb ./data/sample_data/sample_protein.pdb --single_sdf_save_path ./data/sample_data/tmp_sdfs --pocket_save_dir ./data/sample_data/tmp_pockets`

   or use bash command script in bash_scripts dir: You just need to replace the corresponding parameters

   `cd ~/EquiScore/bash_scripts`
   `bash Getpocket.sh`
3. Then, you have all data to predict protein-ligand interaction by EquiScore! Be patient. This is the last step!

   `python Screening.py --ngpu 1 --test --test_path ./data/sample_data/ --test_name tmp_pockets --pred_save_path  ./data/test_results/EquiScore_pred_for_tmp_pockets.csv`

   or use bash command script in bash_scripts dir: You just need to replace the corresponding parameter

   `cd ~/EquiScore/bash_scripts`
   `bash Screening.sh`
4. Until now, you get all prediction result in pred_save_path!

## **Using the provided model weights for evaluation and Reproduces the benchmark result**

   Just like screen compounds for a target, benchmark dataset have many targets for screening, so we implemented a script to calculate the results

### **just some steps needed**

1. We provided Preprocessed pockets on zenodo (download pockets from [zenodo](https://doi.org/10.5281/zenodo.8047224)). IF YOU WANT GET RAW DATASET PLEASE DOWNLOAD RAW DATA FROM REFERENCE PAPERS.
2. you need to download the Preprocessed dataset and extract data to ./data/external_test_data.(for example, all pockets in DEKOIS2.0 docking by Glide SP should be extract into one dir like ./data/external_test_data/dekois2_pocket)
3. if you want to preprocessed data to get pocket , all pocket file names should contain '_active' for active ligand,'_decoy' for decoys and  all pockets in a dir for one benchmark dataset
4. run script (You can use the nohup command and output redirects as you normally like):

   `python Independent_test_dist.py --test --test_path './data/external_test_data' --test_name dekois2_pocket --test_mode multi_pose`

   the result will be saved in ~/EquiScore/workdir/official_weight/

   or use bash command script in bash_scripts dir: You just need to replace the corresponding parameter

   `cd ~/EquiScore/bash_scripts`
   `bash Benchmark_test.sh`

   use **multi_pose** arg if one ligand have multi pose and set pose_num and **idx_style** in args ，see args `--help for more details`

## **Retraining EquiScore 🤖 Model**

### **Retraining EquiScore or fine tune your model is also very simple!**

1. you need to download the traing dataset , and extract pocket data to ./data/training_data/PDBscreen
   (You can also use your own private data, As long as it can fit to EquiScore after processing)
2. use uniprot id to deduplicated data and split data in `./data/data_splits/screen_model/data_split_for_training.py`
   in this script, will help deduplicated dataset by uniprot id and split train/val data and save data path into a pkl file (like "train_keys.pkl, val_keys.pkl, test_keys.pkl").
3. run train.py script:

   `python Train.py --ngpu 1 --train_keys your_keys_path --val_keys your_keys_path --test_keys your_keys_path --test`

   or use bash command script in bash_scripts dir: You just need to replace the corresponding parameter

   `cd ~/EquiScore/bash_scripts`
   `bash Training.sh`

   (**or If you wish to expedite the training process, please refer to the preprocessing workflow in dataset.py, save the data to the LMDB database, and then specify the LMDB path in the training script by adding --lmdb_cache lmdb_cache_path to replace --test like we did in bash command** )

## Citation

   Cao D, Chen G, Jiang J, et al. EquiScore: A generic protein-ligand interaction scoring method integrating physical prior knowledge with data augmentation modeling[J]. bioRxiv, 2023: 2023.06. 18.545464.
   doi: https://doi.org/10.1101/2023.06.18.545464

## License

MIT
