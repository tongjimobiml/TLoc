# TLoc
Transfer Learning-based Outdoor Position Recovery for Telco Data developed on Python2.7 <br>

## Usage
### Data format
MR data: /2g/data_2g.csv <br>
Base station location info: /2g/BS_ALL.csv <br>

### Train a Tranfer learning-based Localization Model
Import **util_tloc** and **TRF** (**The implmentation of Structure transfer**)modules in your code ("TLoc_Example_Code.ipynb"). <br>

Divide the MR data by serving BS, and each serving BS indicates a certain domain. Compute the **domain distance** between each pair of domains.<br>

Train a non-transfer model in each domain. If the prediction error of a ceratin domain (which can be treated as a **target domain**) is greater than a threshold, search source domains for this domain by **domain distance**. <br>

Train a transfer random forest model by training data from source and target domains. <br>


