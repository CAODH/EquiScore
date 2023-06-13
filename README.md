# EquiScore: A generic protein-ligand interaction scoring method integrating physical prior knowledge with data augmentation modeling

### [Paper on arXiv](https://arxiv.org/abs/2210.01776)

Implementation of EquiScore, by Duanhua Cao.

This repository contains all code, instructions and model weights necessary to run the method or to retrain a model.

If you have any question, feel free to open an issue or reach out to us: [caoduanhua@zju.edu](gcorso@mit.edu).

![Alt Text](./figs/model_framework.png)

# Dataset

The files in `data` contain the names for the Uniport ID-based data split.

If you want to train one of our models with the data then:

1. download it from [zenodo](https://zenodo.org/record/6408497)
2. unzip the directory and place it into `data` such that you have the path `data/PDBBind_processed`

## Setup Environment

We will set up the environment using [Anaconda](https://docs.anaconda.com/anaconda/install/index.html). Clone the
current repo

    git clone https://github.com/caoduanhua/EquiScore.git

This is an example for how to set up a working conda environment to run the code (but make sure to use the correct pytorch, pytorch-geometric, cuda versions or cpu only versions):

    conda create --name EquiScore python=3.8
    conda activate EquiScore
    and then install all pkgs, in environment.yml file.

    Fortunately! you slao can setup conda environment by command "conda env create -f environment.yml" and done!

# Screen a compound for your target protein

We implemented a Screening.py python script , to help anyone want to screen compounds from a compound library.

## just some steps need to set

1. Docking compounds with target protein to get docking pose , in this step ,you can use any method to get docking pose(Glide,Vina,Surflex,Gold,LeDock), or you can try a deep learning method.
2. Assume that you have obtained the results of the docking in the previous step. Then, get pocket region and compound pose.

   run script:
   python get_pocket.py --docking_result ./data/sample_data/sample_compounds.sdf --recptor_pdb ./data/sample_data/sample_protein.pdb --single_sdf_save_path ./data/sample_data/tmp_sdfs --pocket_save_dir ./data/sample_data/tmp_pockets
3. Then, you have all data to predict protein-ligand interaction by EquiScore! Be patient. This is the last step!

# Running EquiScore on your own complexes

We support multiple input formats depending on whether you only want to make predictions for a single complex or for many at once.
The protein inputs need to be `.pdb` files or sequences that will be folded with ESMFold. The ligand input can either be a SMILES string or a filetype that RDKit can read like `.sdf` or `.mol2`.

For a single complex: specify the protein with `--protein_path protein.pdb` or `--protein_sequence GIQSYCTPPYSVLQDPPQPVV` and the ligand with `--ligand ligand.sdf` or `--ligand "COc(cc1)ccc1C#N"`

For many complexes: create a csv file with paths to proteins and ligand files or SMILES. It contains as columns `complex_name` (name used to save predictions, can be left empty), `protein_path` (path to `.pdb` file, if empty uses sequence), `ligand_description` (SMILE or file path)  and `protein_sequence` (to fold with ESMFold in case the protein_path is empty).
An example .csv is at `data/protein_ligand_example_csv.csv` and you would use it with `--protein_ligand_csv protein_ligand_example_csv.csv`.

And you are ready to run inference:

    python -m inference --protein_ligand_csv data/protein_ligand_example_csv.csv --out_dir results/user_predictions_small --inference_steps 20 --samples_per_complex 40 --batch_size 10 --actual_steps 18 --no_final_step_noise

When providing the `.pdb` files you can run DiffDock also on CPU, however, if possible, we recommend using a GPU as the model runs significantly faster. Note that the first time you run DiffDock on a device the program will precompute and store in cache look-up tables for SO(2) and SO(3) distributions (typically takes a couple of minutes), this won't be repeated in following runs.

# Retraining DiffDock

Download the data and place it as described in the "Dataset" section above.

### Generate the ESM2 embeddings for the proteins

First run:

    python datasets/pdbbind_lm_embedding_preparation.py

Use the generated file `data/pdbbind_sequences.fasta` to generate the ESM2 language model embeddings using the library https://github.com/facebookresearch/esm by installing their repository and executing the following in their repository:

    python scripts/extract.py esm2_t33_650M_UR50D pdbbind_sequences.fasta embeddings_output --repr_layers 33 --include per_tok --truncation_seq_length 4096

This generates the `embeddings_output` directory which you have to copy into the `data` folder of our repository to have `data/embeddings_output`.
Then run the command:

    python datasets/esm_embeddings_to_pt.py

### Using the provided model weights for evaluation

We first generate the language model embeddings for the testset, then run inference with DiffDock, and then evaluate the files that DiffDock produced:

    python datasets/esm_embedding_preparation.py --protein_ligand_csv data/testset_csv.csv --out_file data/prepared_for_esm_testset.fasta
    git clone https://github.com/facebookresearch/esm
    cd esm
    pip install -e .
    cd ..
    HOME=esm/model_weights python esm/scripts/extract.py esm2_t33_650M_UR50D data/prepared_for_esm_testset.fasta data/esm2_output --repr_layers 33 --include per_tok
    python -m inference --protein_ligand_csv data/testset_csv.csv --out_dir results/user_predictions_testset --inference_steps 20 --samples_per_complex 40 --batch_size 10 --actual_steps 18 --no_final_step_noise
    python evaluate_files.py --results_path results/user_predictions_testset --file_to_exclude rank1.sdf --num_predictions 40

<!--
To predict binding structures using the provided model weights run: 

    python -m evaluate --model_dir workdir/paper_score_model --ckpt best_ema_inference_epoch_model.pt --confidence_ckpt best_model_epoch75.pt --confidence_model_dir workdir/paper_confidence_model --run_name DiffDockInference --inference_steps 20 --split_path data/splits/timesplit_test --samples_per_complex 40 --batch_size 10 --actual_steps 18 --no_final_step_noise

To additionally save the .sdf files of the generated molecules, add the flag `--save_visualisation`
-->

### Training a model yourself and using those weights

Train the large score model:

    python -m train --run_name big_score_model --test_sigma_intervals --esm_embeddings_path data/esm2_3billion_embeddings.pt --log_dir workdir --lr 1e-3 --tr_sigma_min 0.1 --tr_sigma_max 19 --rot_sigma_min 0.03 --rot_sigma_max 1.55 --batch_size 16 --ns 48 --nv 10 --num_conv_layers 6 --dynamic_max_cross --scheduler plateau --scale_by_sigma --dropout 0.1 --remove_hs --c_alpha_max_neighbors 24 --receptor_radius 15 --num_dataloader_workers 1 --cudnn_benchmark --val_inference_freq 5 --num_inference_complexes 500 --use_ema --distance_embed_dim 64 --cross_distance_embed_dim 64 --sigma_embed_dim 64 --scheduler_patience 30 --n_epochs 850

The model weights are saved in the `workdir` directory.

Train a small score model with higher maximum translation sigma that will be used to generate the samples for training the confidence model:

    python -m train --run_name small_score_model --test_sigma_intervals --esm_embeddings_path data/esm2_3billion_embeddings.pt --log_dir workdir --lr 1e-3 --tr_sigma_min 0.1 --tr_sigma_max 34 --rot_sigma_min 0.03 --rot_sigma_max 1.55 --batch_size 16 --ns 24 --nv 6 --num_conv_layers 5 --dynamic_max_cross --scheduler plateau --scale_by_sigma --dropout 0.1 --remove_hs --c_alpha_max_neighbors 24 --receptor_radius 15 --num_dataloader_workers 1 --cudnn_benchmark --val_inference_freq 5 --num_inference_complexes 500 --use_ema --scheduler_patience 30 --n_epochs 300

In practice, you could also likely achieve the same or better results by using the first score model for creating the samples to train the confidence model, but this is what we did in the paper.
The score model used to generate the samples to train the confidence model does not have to be the same as the score model that is used with that confidence model during inference.

Train the confidence model by running the following:

    python -m confidence.confidence_train --original_model_dir workdir/small_score_model --run_name confidence_model --inference_steps 20 --samples_per_complex 7 --batch_size 16 --n_epochs 100 --lr 3e-4 --scheduler_patience 50 --ns 24 --nv 6 --num_conv_layers 5 --dynamic_max_cross --scale_by_sigma --dropout 0.1 --all_atoms --remove_hs --c_alpha_max_neighbors 24 --receptor_radius 15 --esm_embeddings_path data/esm2_3billion_embeddings.pt --main_metric loss --main_metric_goal min --best_model_save_frequency 5 --rmsd_classification_cutoff 2 --cache_creation_id 1 --cache_ids_to_combine 1 2 3 4

first with `--cache_creation_id 1` then `--cache_creation_id 2` etc. up to 4

Now everything is trained and you can run inference with:

    python -m evaluate --model_dir workdir/big_score_model --ckpt best_ema_inference_epoch_model.pt --confidence_ckpt best_model_epoch75.pt --confidence_model_dir workdir/confidence_model --run_name DiffDockInference --inference_steps 20 --split_path data/splits/timesplit_test --samples_per_complex 40 --batch_size 10 --actual_steps 18 --no_final_step_noise

## Citation

    @article{corso2023diffdock,
          title={DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking},
          author = {Corso, Gabriele and Stärk, Hannes and Jing, Bowen and Barzilay, Regina and Jaakkola, Tommi},
          journal={International Conference on Learning Representations (ICLR)},
          year={2023}
    }

## License

MIT

## Acknowledgements

We thank Wei Lu and Rachel Wu for pointing out some issues with the code.

![Alt Text](visualizations/example_6agt_symmetric.gif)
