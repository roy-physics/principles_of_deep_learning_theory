# Principles of Deep Learning Theory - Feature Learning
Basically, these notes are intended for chapters 7-10 of the nice textbook "**The Principles of Deep Learning Theory**" by Sho Yaida, Dan Roberts, and Sho Yaida ([Arxiv link](https://arxiv.org/pdf/2106.10165)). My goal is to write notes on everything and to try to validate things empirically. I would like to deeply understand what "feature learning" **means** in deep learning. 

## Chapter 7: Gradient-Based Learning
Gradient-based learning is the fundamental basis for all modern deep learning methods. An important player in this game is the neural tangent kernel (NTK) which we will explore in detail. It was first explored in a seminal paper [Jacot et al. (2018)](https://arxiv.org/abs/1806.07572) in the context of infinite width networks. This chapter is an introductory one. 

### 7.1: Supervised Learning
We want a network, given a set of inputs $x_\alpha$ and outputs $y_\alpha$ to learn a good function to approximate $z(x_\alpha;\theta)\approx y_\alpha$, where $\alpha$ is the data sample index, not the data dimensionality index. First, we define a loss function:
$$\mathcal{L}(z(x_\alpha;\theta),y_\alpha)$$
A common loss is the mean squared error (MSE) loss:
$$\mathcal{L}_{\rm MSE}(z(x_\alpha;\theta),y_\alpha) = \frac{1}{2}\Big[z(x_\alpha;\theta)-y_\alpha\Big]^2$$
Basically, it's the negative log likelihood of a Gaussian distribution. Ideally, we want to adjust the model parameters $\theta$ to minimise the loss averaged over the entire training data distribution.
$$\langle \mathcal{L}(\theta)\rangle = \int dx \,dy\,p(x,y)\,\mathcal{L}(z(x;\theta),y)$$
In practice, we don't have access to the true distribution of $x,y$, we only have a finite sampling with our training data, so we minimise the following sampling:
$$\mathcal{L}_{\mathcal{A}}(\theta) = \sum_\alpha \mathcal{L}(z(x_\alpha;\theta),y_\alpha)$$
where $\mathcal{A}$ is the training data distribution. What we want is to the find the best $\theta$ to minimise this total training loss:
$$\theta^* = \textrm{arg min}_\theta\, \mathcal{L}_{\mathcal{A}}(\theta)$$
Ideally, we really want our network to pick parameters $\theta^*$ that maximise the true expected loss, not just memorise the sampled training data distribution. This property, of performing well on the general distribution and not just on the training distribution, is called **generalisation**. We always need to be worried about whether the training dataset $\mathcal{A}$ is biased, or has too high variance to properly learn. To this end, we often have a separate dataset $(x_{\dot{\beta}},y_{\dot{\beta}})\in \mathcal{B}$ which is our test set. The model has not seen this data so it's a good way of probing whether the model has generalised or just memorised the training distribution. We will define the loss over the test data as $\mathcal{L}_{\mathcal{B}}$. 

### 7.2: Gradient Descent and Function Approximation
The idea behind gradient descent is just to calculate the gradient of the loss function w.r.t. parameters $\theta_\mu$ and update those parameters incrementally in the direction of decreasing loss i.e. the direction of decreasing gradient. Thus, updates look like the following:
$$\theta_\mu(t+1) = \theta_\mu(t) - \eta \,(\partial_{\theta_\mu}\mathcal{L}_{\mathcal{A}})|_{\theta_\mu(t)}$$
where $\eta$ is the learning rate and $t$ is the learning step. This is pretty simple. 

For sufficiently small learning steps, the loss is guaranteed to decrease (assuming non-zero loss). Let's expand the loss perturbative to see this. 
$$\Delta \mathcal{L} = \mathcal{L}(\theta(t+1)) - \mathcal{L}(\theta(t)) \approx \partial_{\theta_\mu} \mathcal{L}(t)\partial_t \theta_\mu(t) + \mathcal{O}(\eta^2) = -\eta \,(\partial_{\theta_\mu}\mathcal{L}_{\mathcal{A}})(\partial_{\theta_\mu}\mathcal{L}_{\mathcal{A}})\Big|_{\theta(t)}+\mathcal{O}(\eta^2)$$
where we Einstein sum over over the $\mu$ parameter indices. That is quadratic in the loss derivative so the change in loss is negative for sufficiently small $\eta$. 

The idea is to iterate over these small updates to eventually reach a stable minimum of the training loss. However, doing so is challenging and there are several variants of this training algorithm. One extremely popular variance is stochastic gradient descent (SGD) where we update following this algorithm:
$$\theta_\mu(t+1) = \theta_\mu(t) - \eta\sum_{\mathcal{S}_t} \,(\partial_{\theta_\mu}\mathcal{L}_{\mathcal{S}_t})|_{\theta_\mu(t)}$$
where $\mathcal{S}_t \subset \mathcal{A}$ is a **batch** and is a subset of the training data distribution. Batches may get shuffled with each time step or **epoch**, hence the $t$ index on the batch. What's nice about SGD is that the random selection of batches actually provides a regularisation mechanism because the network sees different data samples during every epoch i.e. it can't just learn to optimise the entire training dataset, it has to actually learn good parameters for every random batch. 

Final note: A more general way of writing the gradient update rule is as follows:
$$\theta_\mu(t+1) = \theta_\mu(t) - \eta \,\lambda_{\mu\nu}(\partial_{\theta_\nu}\mathcal{L}_{\mathcal{A}})|_{\theta_\nu(t)}$$
where $\lambda_{\mu\nu}$ is a learning rate tensor that tells us how the gradient w.r.t. parameter $\theta_\nu$ is used to update parameter $\theta_\mu$. For the traditional case, $\lambda_{\mu\nu} = \delta_{\mu\nu}$ (so the identity matrix). We can thus rewrite our change in loss:
$$\Delta \mathcal{L} =  -\eta \,\lambda_{\mu\nu}(\partial_{\theta_\mu}\mathcal{L}_{\mathcal{A}})(\partial_{\theta_\nu}\mathcal{L}_{\mathcal{A}})\Big|_{\theta(t)}+\mathcal{O}(\eta^2)$$
Naturally, we want this tensor $\lambda_{\mu\nu}$ to be positive semi-definite s.t. the loss decreases in general for small learning rates (i.e. the eigenvalues are positive so that each step decreases the loss).
#### Neural Tangent Kernel (NTK)
There is another structure which naturally arises from this optimisation process. Consider the gradient of the loss. 
$$\frac{d\mathcal{L}_{\mathcal{A}}}{d\theta_\mu} = \frac{\partial\mathcal{L}_{\mathcal{A}}}{\partial z_{i;\alpha}}\frac{d z_{i;\alpha}}{d\theta_\mu}$$
where $i$ are the output dimension indices and $\alpha$ are the training dataset sample indices and we are doing an Einstein sum. $z_{i;\alpha}$ is the network output w.r.t data sample input $x_\alpha$. With this chain rule, let's rewrite our change in the loss:
$$\Delta \mathcal{L}_{\mathcal{A}} = -\eta \,\Bigg[\frac{\partial\mathcal{L}_{\mathcal{A}}}{\partial z_{i_1;\alpha_1}}\frac{\partial\mathcal{L}_{\mathcal{A}}}{\partial z_{i_2;\alpha_2}}\Bigg]\Bigg[\lambda_{\mu\nu}\frac{d z_{i_1;\alpha_1}}{d\theta_\mu}\frac{d z_{i_2;\alpha_2}}{d\theta_\nu}\Bigg] + \mathcal{O}(\eta^2)$$
The first quantity is the function approximation error i.e. how sensitive is the loss w.r.t. the NN function outputs. For MSE, this tensor looks like the following:
$$\frac{\partial\mathcal{L}_{\mathcal{A}}}{\partial z_{i;\alpha}} = z_i(x_\alpha;\theta) -  y_{i;\alpha} \equiv \epsilon_{i;\alpha}$$
where we've defined the error tensor $\epsilon_{i\alpha}$. 

The second quantity in the change of the loss is called the neural tangent kernel (NTK). 
$$H_{i_1 i_2;\alpha_1\alpha_2} = \lambda_{\mu\nu}\frac{d z_{i_1;\alpha_1}}{d\theta_\mu}\frac{d z_{i_2;\alpha_2}}{d\theta_\nu}$$
where we sum over the parameter indices. This NTK structure is a function of the output dimension and the datasample indices. This tensor can be computed for the training dataset but also for other datasets. 

The NTK is the main driver of the function-approximation dynamics. To see this, consider a general observable of the model's outputs ${\Theta}(z(x_{\delta_i};\theta),\ldots,z(x_{\delta_M};\theta))$ where $x_{\delta_i}\in \mathcal{D}$ is some general dataset. We can calculate the change of this observable quantity as the network is updated:
$${\Theta}(\theta_{t+1}) - {\Theta}(\theta_t) \approx \frac{\partial {\Theta}}{\partial z_{i;\delta}}\frac{\partial z_{i;\delta}}{\partial \theta_\mu}\frac{\partial \theta_\mu}{\partial t} + \mathcal{O}(\eta^2)$$
$$\approx -\eta\,\frac{\partial {\Theta}}{\partial z_{i;\delta}}\frac{\partial \mathcal{L}}{\partial z_{j;\alpha}}\,\lambda_{\mu\nu}\frac{\partial z_{i;\delta}}{\partial \theta_\mu}\frac{\partial z_{j;\alpha}}{\partial \theta_\nu} + \mathcal{O}(\eta^2)$$
$$\therefore {\Theta}(\theta_{t+1}) - {\Theta}(\theta_t)\approx -\eta\,\frac{\partial {\Theta}}{\partial z_{i;\delta}}\frac{\partial \mathcal{L}}{\partial z_{j;\alpha}}\,H_{ij;\delta\alpha} + \mathcal{O}(\eta^2)$$
where $\delta\in\mathcal{D}$ and $\alpha\in\mathcal{A}$. The NTK basically arises whenever we want to compute the change in a quantity w.r.t. a parameter update. It will thus always involve the training dataset as well as the dataset for which one is evaluating the change on. This comes from the following general formula:
$$\frac{d\theta_\mu}{dt} = -\eta\,\lambda_{\mu\nu}\frac{\partial z_{i;\alpha}}{\partial \theta_\nu}\,\epsilon_{i;\alpha}$$
where $\epsilon$ is the error tensor defined above and $\alpha\in\mathcal{A}$.

Thus, the NTK is important because it gives us insight into the ability for a network output or an observable to change with training or, said another way, for the network to learn something from the training dataset $\mathcal{A}$. If we think about the NTK's indices, the diagonal entries determine the sensitivity of the function w.r.t. a specific input datapoint. The off-diagonal entries, however, show the ability for one datapoint to influence a separate datapoint in terms of the output. The off-diagonal entries determine generalisation.  
