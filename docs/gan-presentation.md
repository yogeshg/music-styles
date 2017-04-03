---
title: "Generative Adversarial Nets"
author: Richard Godden and Yogesh Garg
date: \today
output:
	beamer_presentation:
		slide_level: 2
---

<!--
	tutorial: http://pages.stat.wisc.edu/~yandell/statgen/ucla/Help/Producing%20slide%20shows%20with%20Pandoc.html
	beamer full documentation: http://ctan.math.utah.edu/ctan/tex-archive/macros/latex/contrib/beamer/doc/beameruserguide.pdf
	 * Full documentation is not required since we want pandoc to do most of the heavy lifting
	 * Yet, it may be required to go through sections:
	 * 13: Graphics
	 * 14: Animations
	 * 15: Themese, let's stick with warsaw, but read if interested

        create pdf with pandoc -t beamer gan-presentation.md -V theme:Warsaw -o gan-presentation.pdf
'
-->

# Introduction
## What are Generative models?
 * have a data set of training examples x_i ~ p_data(z)
 * want to be able to generate new examples x_j ~ p_model(z)
 * want that p_model is aproximatel p_data

## Why Generative mode
 * many tasks require it
 * unsupervised learning
 * clean up noisy or missing data
 * reinforcement learning

## Game theory
 * zero sum game
 * equilibrium

## Adversarial networks
 * Game between two players
 * Generator:
 ** tries to create fake samples that look like the real thing
 * Descriminator:
 ** tries to tell which are fake and which are real


# Methods
## Definitions
 * Z some data space
 * X an object (generated or real)
 * Generator G: Z -> X 
 * Descriminator D: X -> {0,1}

## Equations

### Discriminator loss
D tries to identify fakes
$\underset{D}{\operatorname{argmax}}\mathbb{E}_{x~p_{data}(x)}(\log D(G(x))) + \mathbb{E}_{z~p_{z}(z)}(\log(1- D(G(z))))$

### Generator loss
## Algorithms
 * https://www.youtube.com/watch?v=CILzNj2MP3s

# Results
## Theoretical
### convergence
## Experimental
### Mnist
### cfar

# Conclusions
## Advantages
## Disadvantages
### Overfitting to values of the parameters of the distributions
 * Mean of gaussian example https://www.youtube.com/watch?v=mObnwR-u8pc
 * https://www.youtube.com/watch?v=0r3g7-4bMYU

## Interesting experiments
 * Numbers 1 to 5 as shown in paper
 * Faces turn
 * Man with Glasses - Man without glasses + woman = Woman with Glasses

# Further research (may not include)
## Conditional of some input label
## Pretrained gan weights used to improve supervised classification tasks with small datasets
