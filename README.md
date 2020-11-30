Requirements

  To install requirements:

    pip install -r requirements.txt

Repository Structure
  data/:processed data
  
  main.py: The main entrance of the model. You can change dataset name, l2 coefficient(note that for Sub_Flickr the l2_coef =1, and for other datasets, the l2_coef can be smaller value, like 0.01 ) etc in line 239--253.
  
  model.py: The framework of our model
  
  sub_data.py: The code to create social sub-graph data "Sub_Flickr"
  
  tudata.py, mydataset.py : Data preprocessing
  
  gnn_layer.py: The two gnn layers of SGC
  
  chemical.py, scaling_sgc.py, translation_sgc.py: About the scaling and shifting transformation
  
  earlystopping.py: The earlystopping function


Train

  To train the model in the paper:
  
    python main.py




