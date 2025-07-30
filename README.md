
1. Installation:
  - `conda create -n benchmark-vfm-ss python=3.10`
  - `conda activate benchmark-vfm-ss`
  - `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124`

2. Setup: There are scripts for training linear head and full-finetuning with a mask decoder on ADE20K. To run them, first you need to update the `root` folder according to your folder structure. `root` folder should include the ADE20K/PASCAL zip folder. You can use these commands to get them:
  - `wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip`
  - `wget http://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar`, then you have to convert it to zip: 
     - `tar -xf  VOCtrainval_11-May-2012.tar`
     - `zip -r VOCtrainval_11-May-2012.zip VOCdevkit`
     - `rm -rf VOCdevkit`
     - `rm -rf VOCtrainval_11-May-2012.tar`

3. Before running the scripts, you need to convert the ViT models to timm format. You can do it by running `convert_image_vit_ckpt.py` for vanilla ViT/iBOT/CrIBo models, and `convert_object_vit_ckpt.py` for ODIS models. The converted checkpoints will be saved in the same checkpoints folder with an additional `.timm` suffix. For example, for `checkpoint.pth`, you will also have `checkpoint.pth.timm`.

4. Model logs everything to Wandb by default, you can update the settings in the scripts. Please put your Wandb API key in the `~/.wandb_key` file. 

5. Now, you should be able to run the scripts without any problem.