Requirements

  To install requirements:

    pip install -r requirements.txt

Repository Structure
  data/:processed data
  
  main.py: The main entrance of the model. You can change dataset name, l2 coefficient(note that for Sub_Flickr the l2_coef =1, and for other datasets, the l2_coef can be smaller value, like 0.01 ) etc in line 237--248.
  
  model.py: The framework of our model
  
  sub_data.py: The code to create social sub-graph data "Sub_Flickr", 800 small ego-networks,  extracted from an online image sharing social network. Note that if we want to use "Sub_Flickr" as the dataset we don't have the 'data'directory(the processed data), we need to run this file before running main.py. 
  
  tudata.py, mydataset.py : Data preprocessing
  
  gnn_layer.py: The two gnn layers of SGC
  
  chemical.py, scaling_sgc.py, translation_sgc.py: About the scaling and shifting transformation
  
  earlystopping.py: The earlystopping function


Train and test

  To train and test the model in the paper(note that: 1)the following single file includes all the data split into training and testsing set; 2) we just need run this file and all things can be done, including data preprocessing, training and testing):
  
    python main.py




