# D2BGAN Model Implementation from scratch

D2BGAN is a Generative Adversarial Network (GAN) model that is designed to convert low light images to bright images. It is an unpaired GAN-based image enhancement operation that uses cycle consistency, geometric consistency, and illumination consistency. The model has been shown to provide competitive results on standard benchmark datasets, and it has been observed to perform well on DICM, LIME, and MEF datasets when D2BGAN was applied. However, it does not perform well on backlit images.

# D2BGAN Model Architecture

![D2BGAN Model Architecture](/images/D2BGAN_architecture.png)

# D2BGAN Dataflow 

![D2BGAN Dataflow](/images/D2BGAN_data_flow.png)

# D2BGAN Loss contribution

![D2BGAN Loss contribution](/images/D2BGAN_loss_contribution.png)

## References

[D2BGAN](https://arts.units.it/retrieve/e2913fdf-656a-f688-e053-3705fe0a67e0/D2BGAN_A_Dark_to_Bright_Image_Conversion_Model_for_Quality_Enhancement_and_Analysis_Tasks_Without_Paired_Supervision.pdf)