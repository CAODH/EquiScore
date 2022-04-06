# train.py

1. write a json file like default in your json file (default = /home/caoduanhua/score_function/GNN/GNN_graphformer_pyg/config_files/train.json)
2. make train/val data by divide.py (in a new dir))
3. **fill params in your new json file**
4. run train.py in linux :

   cd to ~/score_function/GNN/GNN_graphformer_pyg

   nohup python train.py --json_path > log_file.log&
5. independent mode

   nohup python independent_test.py --json_path  --best_name "model to test path" > log_file.log&

   or you can fill some params in independent_test.py!
