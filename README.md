
# SkyNet — Diffusion Models

This repository contains our term project for **UMC 203: Introduction to AI and ML** at the **Indian Institute of Science (IISc)**. The project focuses on a systematic study of **diffusion models**, a class of generative models that have recently emerged as a dominant paradigm in generative modeling.

## 1. Introduction

Diffusion models constitute a class of generative models that learn to reverse a gradual noising process, transforming random (typically Gaussian) noise into structured data through iterative denoising. In recent years, they have achieved state-of-the-art performance in image synthesis while maintaining stable training dynamics compared to earlier approaches such as Variational Auto-Encoders (VAEs). They also admit a unifying interpretation as score-based generative models, where the reverse diffusion process corresponds to estimating the score (gradient of log-density) of the data distribution.

In this project, we explore the key ideas underlying diffusion models through both **theoretical analysis and empirical validation**. Our study focuses on:

- Understanding DDPM from multiple perspectives and design choices  
- Comparing DDPM and DDIM  
- Exploring equivalence with score-based models  
- Investigating guided diffusion techniques  

Together, these experiments provide a coherent view of the landscape of diffusion-based generative modeling. :contentReference[oaicite:0]{index=0}



## 2. Background

### 2.1 Denoising Diffusion Probabilistic Models (DDPM)

DDPMs are formulated as discrete-time Markov chains where the forward process incrementally adds Gaussian noise according to a variance schedule:

- The forward process gradually transforms data into noise.
- As the number of steps increases, the distribution converges to a standard Gaussian.
- A neural network is trained to predict the noise added at each timestep.

The reverse process is learned to invert this noising procedure, enabling generation from pure noise. However, sampling requires iterating through all timesteps, making it computationally expensive.



### 2.2 Denoising Diffusion Implicit Models (DDIM)

DDIM generalizes the diffusion framework by introducing a **non-Markovian process** that preserves the same marginal distributions as DDPM.

Key properties:
- Enables **deterministic sampling**
- Allows **skipping timesteps**

Thus, DDIM provides a trade-off between computational efficiency and stochastic diversity while retaining comparable sample quality.



## 3. Experiments

### 3.1 Comparison with VAEs

We evaluate the claim that diffusion models outperform VAEs in image generation. Using the **Fréchet Inception Distance (FID)** as the evaluation metric, we observe that:

- DDPM consistently achieves lower FID than VAE across epochs  
- This supports the claim of superior sample quality  



### 3.2 Faster Sampling with DDIM

While DDPM requires sequential sampling across all timesteps, DDIM allows sampling over a subset of steps.

Observations:
- DDIM achieves significantly **lower sampling time**
- Maintains **comparable KID scores**


### 3.3 Deterministic Nature of DDIM

A key distinction between DDPM and DDIM lies in stochasticity:

- DDPM: stochastic reverse process → diverse outputs  
- DDIM: deterministic reverse process → identical outputs for same initialization  


### 3.4 Effect of Number of Timesteps (T)

We analyze how the number of diffusion steps affects both the latent distribution and generated samples.

#### Convergence of Latent Distribution

- As \( T \) increases, the latent variable approaches a Gaussian distribution  
- KL divergence decreases monotonically, indicating improved convergence  

#### Effect on Sample Quality

- Increasing sampling steps improves image quality  
- Observed through decreasing FID values  


### 3.5 Effect of Noise Schedule

We compare different variance schedules:

- Linear  
- Cosine  
- Exponential  

Findings:
- Exponential schedule performs significantly worse  
- Linear and cosine perform comparably, with cosine slightly better  
- Performance depends on how evenly noise is distributed across timesteps  


### 3.6 Equivalence with Score-Based Models

We study the relationship between diffusion models and score-based generative models.

- The DDPM objective is equivalent to **denoising score matching**  
- Sampling can be interpreted as solving a **reverse-time stochastic differential equation (SDE)**  

Empirical validation shows:
- Similar sampling trajectories  
- Comparable distribution metrics (MMD)  


### 3.7 Guided Diffusion (Classifier Guidance)

We implement classifier-guided diffusion to generate conditional samples.

- A classifier is trained on noisy inputs  
- Gradients from the classifier guide the sampling process  
- Enables generation conditioned on labels  

Example: generating specific MNIST digits using guidance.


## 4. Conclusion

Through this project, we systematically investigated both the theoretical foundations and practical implementations of diffusion models.

Key takeaways:
- Diffusion models outperform traditional generative approaches like VAEs in image quality.  
- DDIM provides a principled method for faster sampling  
- Model performance depends critically on timestep count and noise schedules.
- Diffusion models are closely connected to score-based methods and SDE formulations.
- Guided diffusion enables controllable generation.

Overall, this work provides a comprehensive understanding of diffusion-based generative modeling, implementation, and experimentation.


## Datasets

- MNIST  
- CIFAR-10  
- Fashion-MNIST  
- CelebA  



## Implementation

- U-Net based architecture with:
  - Residual blocks  
  - Group normalization  
  - Self-attention  
  - Time-step conditioning  
- Training:
  - Mean squared error loss (noise prediction)  
  - Adam optimizer  
  - Exponential Moving Average (EMA)  



## References

1. Jonathan Ho, Ajay Jain, and Pieter Abbeel.  
   *Denoising Diffusion Probabilistic Models*, 2020.  
   https://arxiv.org/abs/2006.11239  

2. Jiaming Song, Chenlin Meng, and Stefano Ermon.  
   *Denoising Diffusion Implicit Models*, 2022.  
   https://arxiv.org/abs/2010.02502  

3. Diederik P. Kingma and Max Welling.  
   *Auto-Encoding Variational Bayes*, 2022.  
   https://arxiv.org/abs/1312.6114  

4. Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole.  
   *Score-Based Generative Modeling through Stochastic Differential Equations*, 2021.  
   https://arxiv.org/abs/2011.13456  

5. Prafulla Dhariwal and Alex Nichol.  
   *Diffusion Models Beat GANs on Image Synthesis*, 2021.  
   https://arxiv.org/abs/2105.05233  

6. Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli.  
   *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*, 2015.  
   https://arxiv.org/abs/1503.03585  

7. Alex Nichol and Prafulla Dhariwal.  
   *Improved Denoising Diffusion Probabilistic Models*, 2021.  
   https://arxiv.org/abs/2102.09672  

8. Davide Carbone.  
   *Hitchhiker’s Guide on the Relation of Energy-Based Models with Other Generative Models, Sampling and Statistical Physics: A Comprehensive Review*, 2025.  
   https://arxiv.org/abs/2406.13661  

9. Chieh-Hsin Lai, Yang Song, Dongjun Kim, Yuki Mitsufuji, and Stefano Ermon.  
   *The Principles of Diffusion Models*, 2025.  
   https://arxiv.org/abs/2510.21890  

10. Stanley H. Chan.  
    *Tutorial on Diffusion Models for Imaging and Vision*, 2025.  
    https://arxiv.org/abs/2403.18103   

## Acknowledgments
We are grateful to our mentor, Sahil Chaudhary, for his immense guidance and support, which played a crucial role in shaping this work. We also thank Professor Aditya Gopalan and Professor Shishir N.Y. Kolathaya for this learning opportunity.


## Authors
- [Aditey Nandan](https://in.linkedin.com/in/aditey-nandan)
- [Avani Lakshmi Udupa]()
- [Durga Naniwadekar]()
- [Jashandeep Thind]()
- [Liu Runbang]()