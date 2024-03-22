# D2BGAN Model Implementation from scratch

![Banner](https://user-images.githubusercontent.com/73076876/138980624-cbcf98bc-ac43-41a5-a399-5ca186858be0.png)

# Paper 


### D2BGAN: A Dark to Bright Image Conversion Model for Quality Enhancement and Analysis Tasks Without Paired Supervision

[D2BGAN Paper](https://ieeexplore.ieee.org/document/9784432)

# Summary of D2BGAN paper

D2BGAN is a Generative Adversarial Network (GAN) model that is designed to convert low light images to bright images. It is an unpaired GAN-based image enhancement operation that uses cycle consistency, geometric consistency, and illumination consistency. The model has been shown to provide competitive results on standard benchmark datasets, and it has been observed to perform well on DICM, LIME, and MEF datasets when D2BGAN was applied. However, it does not perform well on backlit images.


# D2BGAN Model Architecture

![D2BGAN Model Architecture](/images/D2BGAN_architecture.png)

# D2BGAN Dataflow 

![D2BGAN Dataflow](/images/D2BGAN_data_flow.png)

# D2BGAN Loss contribution

![D2BGAN Loss contribution](/images/D2BGAN_loss_contribution.png)

# D2BGAN Experimental Results

The following are the experimental results of the D2BGAN model by the original authors.

![D2BGAN Experimental results](images/experimental_results.gif)

## References

[D2BGAN](https://arts.units.it/retrieve/e2913fdf-656a-f688-e053-3705fe0a67e0/D2BGAN_A_Dark_to_Bright_Image_Conversion_Model_for_Quality_Enhancement_and_Analysis_Tasks_Without_Paired_Supervision.pdf)