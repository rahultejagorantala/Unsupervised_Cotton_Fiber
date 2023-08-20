# AutoEncoder 

This repo consits of code modules of an AutoEncoder. The main aim of this project is to use the embeddings coming out of the encoder module and perform clustering.

Run the following command to start training and then subsequently extract emnbeddings from the encoder and then perform K-Means Clustering.
```shell
python AutoEncoder.py --config_exp encoder.yaml
```

Below are the examples of reconstructed images in DTD dataset. After training clustering is performed on the embeddings and an accuracy of 66%.

<table>
  <tr> 
    <td align="center"><img src="https://github.com/rahultejagorantala/Unsupervised_Cotton_Fiber/blob/main/AutoEncoder/Images/epoch%20-%200.png" width=300 height=100 ></td>
    <td align="center"><img src="https://github.com/rahultejagorantala/Unsupervised_Cotton_Fiber/blob/main/AutoEncoder/Images/epoch%20-%201000.png" width=300 height=100 ></td>
    <td align="center"><img src="https://github.com/rahultejagorantala/Unsupervised_Cotton_Fiber/blob/main/AutoEncoder/Images/epoch%20-%202000.png" width=300 height=100 ></td>
  </tr>
 </table>
 
