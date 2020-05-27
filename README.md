# MDS Capstone Project
## DCGAN for Julia
This project is a DCGAN package base on Flux.jl machine learning library. 

## Why we choose DCGAN
 
Orginal GAN is hard to train.   
Stabilize Generative Adversarial networks with some architectural constraints.  
Most used architecture.  

### paper  
[[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks]](https://arxiv.org/pdf/1511.06434.pdf)  

**Architecture guidelines for stable Deep Convolutional GANs**  

* Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).  
* Use batchnorm in both the generator and the discriminator  
* Remove fully connected hidden layers for deeper architectures. Just use average pooling at the end.  
* Use ReLU activation in generator for all layers except for the output, which uses Tanh.
* Use LeakyReLU activation in the discriminator for all layers.  
## How to train a GAN model?

The typical GAN model training steps are as follows:

1. Specify input for your network, such as random noise, or input picture 
2. Build two models: generator and discriminator network.
3. Define loss function。
4. Use Flux.jl to set the training optimizer.
5. Start training

## Structure
> Generator:
``` 
function Generator()
    return Chain(
        Dense(noise_dim, 7 * 7 * 256; ),
        BatchNorm(7 * 7 * 256, relu),
        x->reshape(x, 7, 7, 256, :),
        ConvTranspose((5, 5), 256 => 128; , stride = 1, pad = 2), BatchNorm(128, relu),
        ConvTranspose((4, 4), 128 => 64; , stride = 2, pad = 1), BatchNorm(64, relu),
        ConvTranspose((4, 4), 64 => 1, tanh; , stride = 2, pad = 1),
        )
end
```
> Discriminator:
```
function Discriminator()
    return Chain(
        Conv((4, 4), channels => 64, leakyrelu; stride = 2, pad = 1), Dropout(0.25),
        Conv((4, 4), 64 => 128, leakyrelu; stride = 2, pad = 1), Dropout(0.25), x->reshape(x, 7 * 7 * 128, :),
        Dense(7 * 7 * 128, 1; ))
end
```
> G Loss:   
>
		loss = mean(logitbinarycrossentropy.(fake_output, 1f0))
> D Loss:   
>
		real_loss = mean(logitbinarycrossentropy.(real_output, 1f0))
        fake_loss = mean(logitbinarycrossentropy.(fake_output, 0f0))
        loss = 0.5f0 * (real_loss +  fake_loss)
> Solver: Adam lr=0.0002    stepG:stepD=1:1   
> Data: MNIST __normalized to [-1,1] for tanh.


## Usage
### Example 

Unconditional MNIST image generation

Install required packages:
```
using Pkg; 
Pkg.add("Flux")
Pkg.add("Images")
Pkg.add("Statistics")
Pkg.add("Printf")
```

## Run this package

```
main()
```

## Result
![pic1](https://github.com/YaoLaplace/Julia-Package/blob/master/DCGAN_result/steps_004600.png "Generate image")





