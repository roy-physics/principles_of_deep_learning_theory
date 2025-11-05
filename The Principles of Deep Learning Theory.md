# The Principles of Deep Learning Theory
*An Effective Theory Approach to Understanding Neural Networks*

*Authors*: Daniel A. Roberts, Sho Yaida, Boris Hanin
Arxiv Link: [link](https://arxiv.org/pdf/2106.10165)
## Chapter 0: Initialisation

Much of modern successes of machine learning come from deep learning and representation learning that transforms data into increasingly refined forms helpful for extremely nonlinear tasks. 

Deep learning theory is still in its infancy, large disconnect between theory and practice (practitioners are blazing ahead, far beyond current theoretical understanding). 

### 0.1 An Effective Theory Approach
*"Steam navigation brings nearer together the most distant nations. . . . their theory is very little understood, and the attempts to improve them are still directed almost by chance. . . . We propose now to submit these questions to a deliberate examination." - Sadi Carnot*

Theoretical physics offers a method of finding simple effective theories of complex systems with large number of components, from thermodynamics to fluids to particle physics, etc.

We need for deep learning what statistical mechanics was to steam engines or what quantum field theory was for particle physics. 

### 0.2 The Theoretical Minimum
*"The method is more important than the discovery, because the correct method of research will lead to new, even more valuable discoveries." - Lev Landau*

- $f(x)$: Our real target function
- $f(x;\theta)$: Our neural network (NN) parametrised by parameters $\theta$. 
- $p(\theta)$: The probability distribution from which we sample our $\theta$ at initialisation
- $f(x;\theta^*)\approx f(x)$: $\theta^*$ is a parametrisation which makes our NN closely approximate the true function. We adjust $\theta \rightarrow \theta^*$ via a learning algorithm. Note: There may be several different choices of $\theta^*$ which give us a close function approximation.
- $(x,f(x))$: Training data of input, target pairs. The data we use to update the parameters $\theta$. 

**Goal: Understand the distribution of trained NN functions $f(x;\theta^*)$**

To anticipate some of the difficulties with this approach, let's imagine taylor expanding our NN about the initial set of parameters $\theta$. 
$$f(x;\theta^*)\approx f(x;\theta) + (\theta^*-\theta)_i\cdot \partial_{\theta_i}f(x;\theta) + \frac{1}{2}(\theta^*-\theta)_i(\theta^*-\theta)_j\cdot \partial_{\theta_i}\partial_{\theta_j}f(x;\theta) + \ldots$$
- *Issue 1*: There are infinitely many terms in this expansion. How many terms needed to get a good approximation of our final trained network?
- *Issue 2*: Our randomly sampled $\theta$ map to an infinite number of initial random functions $f,\partial_{\theta} f, \partial^2f/\partial_\theta^2,\ldots$ There's no guarantee that this space is analytically tractable. 
- *Issue 3*: The learned value of the final parameters $\theta^*$ are not unique and depend in a complex way on the initialisation, the learning algorithm, training data, etc. $$\theta^* \equiv [\theta^*](\theta,f,\partial_{\theta}f,\partial^2f/\partial_{\theta}^2,\ldots|\textrm{learning algorithm; training data})$$
If we are somehow able to calculate this dependence, we can in principle derive the final distribution of trained NNs. 
$$p(f^*) \equiv p(f(x;\theta^*|\textrm{learning algorithm; training data}))$$

#### A Principle of Sparsity & the Big Punchline
There are two important quantities when characterising fully connected networks: The depth $L$ and the width $n$. There are two interesting limits:
- Increase depth $L$, keeping width $n$ fixed: The infinite depth limit.
- Increase width $n$, keeping depth $L$ fixed: The infinite width limit.

Turns out that things will simplify significantly in the infinite width limit but not in the infinite depth limit. Let's consider the infinite width limit:
$$\lim_{n\rightarrow \infty} p(f^*)$$
Here's what will happen in this limit:
- All higher order derivatives in our Taylor expansion will become zero: $d^kf/d\theta^k=0$ for $k\geq 2$. We will only need two terms $f, df/d\theta$. 
- The distributions of trained functions and their derivatives will be independent: $$\lim_{n\rightarrow \infty} p(f,df/d\theta, \ldots)=p(f)p(df/d\theta)$$
- The training dynamics become linear and independent of the learning algorithm details, allowing us analytically calculate $\theta^*[\theta,f,df/d\theta; \textrm{training data}]$

It will turn out that the infinite width limit gives us trained neural network distribution function which is totally Gaussian. This comes from the principle of sparsity or the large $N$ limit where infinite $N$ obscures correlations between parameters and signals from complicated patterns of neurons. This causes sparsity in their descriptions. 

The infinite width limit is a poor description of neural networks in reality. Firstly, real NNs are not infinitely wide (not physical) and also, they will show that the representations that NNs learn in the infinite width limit are essentially unchanged from the randomly initialised representations: representation/feature learning is one of the true successes of modern NNs so the lack of feature learning is a problem. 

One way then to describe modern NNs is thus to expand in $1/n$:
$$p(f^*) \approx p^{(0)}(f^*) + \frac{p^{(1)}(f^*)}{n} + \frac{p^{(2)}(f^*)}{n^2}+\ldots$$
This textbook will compute the first order term, truncating at $\mathcal{O}(1/n^2)$. In this "*interacting theory*", our issues are more tractable:
- We will only need to keep track of these terms to compute the $\mathcal{O}(1/n)$ terms: $f, df/d\theta, d^2f/d\theta^2, d^3f/d\theta^3$. 
- The probability distributions of these functions and their derivatives will be tractable analytically. 
- We will be able to use perturbation theory to calculate a closed form for $\theta^*[\theta,f,df/d\theta; \textrm{training data}]$. 

We will get a nearly-Gaussian distribution, just like in QFT one uses perturbation theory to get nearly free-field theories. This expansion will actually demonstrate nontrivial representation/feature learning, just like real NNs. This will be useful as a minimal model for basic deep learning. 

Here are the main punchlines in different limits of $r=L/n$ (width ratio):
- $r\rightarrow 0$: Interactions between neurons turn off. These networks are not technically "deep" though.
- $0 < r \ll 1$: Nontrivial interactions between neurons. These networks are effectively deep. 
- $r\gg 1$: Neurons are very strongly coupled and the NN is overly deep. There are large fluctuations from instantiation to instantiation.


## Chapter 1: Pretraining
*"My strongest memory of the class is th every beginning, when he started, not with some deep principle of nature, or some experiment, but with a review of Gaussian integrals. Clearly, there was some calculating to be done. - Joe Polchinski, reminiscing about Feynman's quantum mechanics class"*

This chapter is about computing correlation functions and Gaussian integrals. It will form the basis of the rest of the book. This is a lovely chapter and also nice for students of quantum field theory. Skip this chapter if you're already an expert at these kinds of calculations.

### 1.1 Gaussian Integrals

#### Single-variable Gaussian Integrals
Consider $e^{-z^2/2}$. Without getting into the derivation into too much detail, we have the following identity:
$$I_1 = \int_{-\infty}^\infty dz\, e^{-z^2/2} = \sqrt{2\pi}$$
This gives us the following standard normal PDF. 
$$p(z) = \frac{1}{\sqrt{2\pi}}e^{-z^2/2}$$
Extend this to a Gaussian with variance $K>0$. 
$$I_K = \int_{-\infty}^\infty dz\, e^{-z^2/2K} = \sqrt{K}\int_{-\infty}^\infty du\, e^{-u^2/2} = \sqrt{2\pi K}$$
Thus, we get the general PDF. 
$$p(z) = \frac{1}{\sqrt{2\pi K}}e^{-z^2/2K}$$
Let's generalise this further but by adding a shift or a mean. 
$$p(z) = \frac{1}{\sqrt{2K}}e^{-(z-s)^2/2K}$$
$s$ is the mean of the distribution since the following identity holds:
$$\mathbb{E}[z] \equiv \langle z\rangle = \int_{-\infty}^\infty dz\,z \,p(z) = s$$
Let's define the expectation value of an observable $\mathcal{O}(z)$:
$$\mathbb{E}[\mathcal{O}(z)] \equiv \langle\mathcal{O}(z) \rangle = \int_{-\infty}^\infty dz\,\mathcal{O}(z)\,p(z)$$
Now, let's define some moments:
$$\langle z^M\rangle = \frac{1}{\sqrt{2\pi K}}\int dx\, e^{-z^2/2K}z^M$$
Naturally, this goes to zero for odd $M$. Thus, let's focus on even integers $M = 2m$ and use the typical Feynman integral trick. 
$$I_{K,m} = \langle z^{2m}\rangle = \int\,dz\,e^{-z^2/2K}z^{2m}=(2K^2\frac{d}{dK})^m \, I_K = (2K^2\frac{d}{dK})^m \, \sqrt{2\pi K}$$
$$I_{K,m} = \langle z^{2m}\rangle = \sqrt{2\pi}\, K^{\frac{2m+1}{2}}\,(2m-1)\cdot(2m-3)\ldots\cdot1$$
Thus, we have our moment expression:
$$\langle z^{2m}\rangle = K^m \,(2m-1)!! = K^m\,\frac{(2m)!}{2^m\,m!}$$
where $(2m-1)!! = (2m-1)\cdot(2m-3)\cdot\ldots\cdot 1 = \frac{(2m)!}{2^m\,m!}$. This is Wick's theorem in 1D. 

There is an alternate, equally nice way of calculating these moments which works nicely when going to multivariable Gaussian distributions. It involves using a source term $J$:
$$Z_{K,J} = \int_{-\infty}^\infty dz\,e^{-z^2/2K + Jz}$$
Evidently, $Z_{K,0} = I_K$. This object is the partition function with a source term, and is a generating function for moments. Let's first complete the square for the exponent:
$$-\frac{z^2}{2K} + Jz = - \frac{(z-KJ)^2}{2K}+ \frac{K J^2}{2}$$
Thus, we can evaluate our integral:
$$Z_{K,J} = e^{KJ^2/2}\int_{-\infty}^\infty dz\,e^{-(z-KJ)^2/2K} = e^{KJ^2/2}\,\sqrt{2\pi K}$$
Nice! Now, notice the following about the moments. 
$$\sqrt{2\pi K}\,\langle z^{2m}\rangle =I_{K,m} = \int dz\,e^{-z^2/2K}z^{2m} = [\,(\frac{d}{dJ})^{2m}\int dz\,e^{-z^2/2K + Jz}]\,|_{J=0} = [\,(\frac{d}{dJ})^{2m}Z_{K,J}\,|_{J=0}$$
We can just calculate that analytically. 
$$\langle z^{2m}\rangle = [\,(\frac{d}{dJ})^{2m}e^{KJ^2/2}]\,|_{J=0}= [\,(\frac{d}{dJ})^{2m}\cdot\sum_n \frac{K^n}{2^n n!}J^{2n}]\,|_{J=0} = K^m\, (2m-1)!! = K^m\frac{(2m)!}{2^m m!}$$
This is exactly the same as our previous result, but derived using this source term formalism (I've skipped a bunch of steps here, but they can be quickly derived). With the 1D case out of the way, let's proceed to multi-dimensional Gaussians!

#### Multivariable Gaussian Integrals
Now imagine multivariable Gaussian for an $N$ dimensional variable $z_\mu$. We have the following basic form (where I use Einstein notation):
$$\exp[\,-\frac{1}{2}z^\mu (K^{-1})_{\mu\nu}z^\nu\,]$$
The covariance matrix is defined as follows:
$$(K^{-1})_{\mu\rho}K^{\rho\nu} = \delta_\mu^\nu$$
where again, I'm computing an Einstein summation. Let's now compute the normalisation factor for our PDF. 
$$I_K = \int d^Nz\,\exp[\,-\frac{1}{2}z^\mu (K^{-1})_{\mu\nu}z^\nu\,] = (\Pi_{i=1}^N \int dz_i)\,\exp[\,-\frac{1}{2}z^\mu (K^{-1})_{\mu\nu}z^\nu\,]$$
Let's diagonalise our matrix $\tilde{K} = (O\cdot K\cdot O^T)_{\mu\nu} = \lambda_\mu\delta^\mu_\nu$, where I'm technically not doing an Einstein summation here. It's just a diagonal $N\times N$ matrix with eigenvalues $\lambda_\mu$. Note also that these change of basis matrices must be orthogonal and thus $(O^T\cdot O)_{\mu\nu} = (O\cdot O^T)_{\mu\nu} = \delta_{\mu\nu}$. With this, we can easily calculate our integral by inserting factors of $O\cdot O^T$ in between $z_\mu$ and $K^{-1}$. 
$$I_K = (\Pi_{i=1}^N \int dz_i\, \exp[\,-\frac{1}{2\lambda_i}(Oz)_i^2\,]) \equiv (\Pi_{i=1}^N \int du_i\, \exp[\,-\frac{1}{2\lambda_i}u_i^2\,]) = \Pi_{\mu=1}^N \sqrt{2\pi \lambda_\mu}$$
This is just the square root of the determinant of our kernel matrix $K$ multiplied by $2\pi$. 
$$I_K = \sqrt{|2\pi K|}$$
Thus, our PDF is pretty simple. 
$$p(z) = \frac{1}{\sqrt{|2\pi K|}}\,\exp[\,-\frac{1}{2}z^\mu (K^{-1})_{\mu\nu}\,z^\nu\,]$$
At this point, Sho and Dan introduce the notation that $K^{-1}_{\mu\nu}\equiv K^{\mu\nu}$, so upper indices mean inverse. **I guess, whatever, this is basically a metric tensor-like quantity and thus upper indices are distinguished from lower indices. In that sense, fair enough. But this isn't really like GR in the sense that $z_\mu = K_{\mu\nu}z^\nu$. Instead, $z_\mu = \delta^\nu_\mu z_\nu$. Or am I getting something fundamentally wrong??? Need to think more.** 
$$K^{\mu\rho}K_{\rho\nu} = \delta^\mu_\nu$$
Thus, we can write the Gaussian in a slightly cleaner form:
$$p(z) = \frac{1}{\sqrt{|2\pi K|}}\,\exp[\,-\frac{1}{2}z_\mu K^{\mu\nu}\,z_\nu\,]$$
Now, if we introduce a slightly shift in the distribution via a mean $s_\mu$, we get the following general form.
$$p(z) = \frac{1}{\sqrt{|2\pi K|}}\,\exp[\,-\frac{1}{2}(z-s)_\mu K^{\mu\nu}\,(z-s)_\nu\,]$$
Let's consider moments of the mean-zero multivariable Gaussian:
$$\langle z_{\mu_1}\ldots z_{\mu_M}\rangle = \frac{1}{\sqrt{|2\pi K|}}\int d^Nz\,\exp[\,-\frac{1}{2}z_\mu K^{\mu\nu}\,z_\nu\,]\,z_{\mu_1}\ldots z_{\mu_M}= \frac{I_{K,(\mu_1,\ldots,\mu_M)}}{I_K}$$
We can solve this by thinking about a generating function. 
$$Z_{K,J} = \int d^Nz\,\exp[\,-\frac{1}{2}z_\mu K^{\mu\nu}\,z_\nu \,+\,J^\mu z_\mu\,]$$
Similar to the 1D case, we can use this notation to compute moments.
$$[\,\frac{d}{dJ^{\mu_1}}\ldots\frac{d}{dJ^{\mu_M}}Z_{K,J}\,]|_{J=0} = \int d^Nz\,\exp[\,-\frac{1}{2}z_\mu K^{\mu\nu}\,z_\nu\,]\,z_{\mu_1}\ldots z_{\mu_M} = I_{K,(\mu_1,\ldots,\mu_M)}$$
We can complete the square for these tensors quantities in the exponent:
$$-\frac{1}{2}z_\mu K^{\mu\nu}\,z_\nu \,+\,J^\mu z_\mu = -\frac{1}{2}K^{\mu\nu}(\,z_\mu - K_{\mu\alpha}J^\alpha\,)(\,z_\nu - K_{\nu\beta}J^\beta\,) + \frac{1}{2}J^\mu K_{\mu\nu}J^\nu$$
$$\equiv -\frac{1}{2}w_\mu K^{\mu\nu} w_\nu + \frac{1}{2}J^\mu K_{\mu\nu}J^\nu$$
where we've defined shifted variables $w_\mu = z_\mu - K_{\mu\alpha}J^\alpha$. We can do our integral pretty easily now. 
$$Z_{K,J} = \exp(\,\frac{1}{2}J^\mu K_{\mu\nu}J^\nu\,)\int d^N w\, \exp(\,-\frac{1}{2}w_\mu K^{\mu\nu}w_\nu\,)$$
$$Z_{K,J} = \sqrt{|2\pi K|}\,\exp(\,\frac{1}{2}J^\mu K_{\mu\nu}J^\nu\,)$$
We can now take derivatives of this function to derive our moments. Odd moments naturally go to zero, so we'll focus on even moments.
$$\langle z_{\mu_1}\ldots z_{\mu_{2m}}\rangle = \frac{1}{I_K}[\,\frac{d}{dJ^{\mu_1}}\ldots\frac{d}{dJ^{\mu_{2m}}}Z_{K,J}\,]|_{J=0}$$
$$ = \frac{1}{2^m\,m!}[\,\frac{d}{dJ^{\mu_1}}\ldots\frac{d}{dJ^{\mu_{2m}}}(\,J^\mu K_{\mu\nu}J^\nu\,)^{2m}\,]|_{J=0}$$
Let's evaluate this expression for $m=1$. 
$$\langle z_{\mu_1} z_{\mu_1} \rangle = \frac{1}{2}[\,K_{{\mu_1}\mu_2} + K_{\mu_2\mu_1}\,] = K_{\mu_1\mu_2}$$
Basically, there are two ways to map the combination $\mu_1,\mu_2$ to $\mu\nu$.  That cancels out the factor of $1/2^1$. Let's now look at $m=2$. 
$$\langle z_{\mu_1} z_{\mu_2} z_{\mu_3} z_{\mu_4}\rangle = \frac{1}{2^2\cdot 2!}[\,\frac{d}{dJ^{\mu_1}}\ldots\frac{d}{dJ^{\mu_{4}}}(\,J^\mu K_{\mu\nu}J^\nu\,)(\,J^\alpha K_{\alpha\beta}J^\beta\,)\,]|_{J=0}$$
$$ = K_{\mu_1 \mu_2}K_{\mu_3 \mu_4} + K_{\mu_1 \mu_3}K_{\mu_2 \mu_4} + K_{\mu_1 \mu_4}K_{\mu_2 \mu_3}$$
In terms of the factors, consider the term $K_{\mu_1 \mu_2}K_{\mu_3 \mu_4}$. There are two ways of mapping $\mu_1,\mu_2$ to $\mu\nu$ and two ways of mapping $\mu_3,\mu_4$ to $\alpha\beta$, which gives us a total factor of $2^2$. We also have a factor of $2!$ which comes from there being two possible index pairs, $\alpha\beta$ and $\mu\nu$, from which to match $\mu_1,\mu_2$ and then one remaining pair for $\mu_3,\mu_4$. That gives us a total factor of $2^2\cdot 2!$, which exactly cancels the denominator. The same holds for the other two index combinations. 

So, we see a clear pattern. For each permutation/index combination, we get a symmetry factor of $2^m \cdot m!$ which will cancel the denominator. We then need to add all the combinations/permutations. 

$$\langle z_{\mu_1}\ldots z_{\mu_{2m}} \rangle = \sum_{\textrm{all pairings}}K_{\mu_{k_1}\mu_{k_2}}\ldots K_{\mu_{k_{2m-1}}\mu_{k_{2m}}}$$
This is the famous Wick contraction formula that is so often used in quantum field theory (we'll discuss the connections in a bit). All the indices are contracted in all possible permutations. 

### 1.2 Probability, Correlation, Statistics, and All That
Everything in the last subsection was for Gaussian PDFs. Let's generalise some things to general PDFs. Consider a probability distribution $p(z)$ of a $N$-dimensional random variable $z_\mu$. We can define the following observable.
$$\langle \mathcal{O}(z)\rangle = \int d^N z\,p(z)\mathcal{O}(z)$$
What can we infer about $p(z)$ given an expected observable $\langle\mathcal{O}(z)\rangle$? Consider the class of moments or M-point correlators:
$$\langle z_{\mu_1}\ldots z_{\mu_M}\rangle = \int d^N z\,p(z)\,z_{\mu_1}\ldots z_{\mu_M}$$
With an infinite series of M-point correlators, we can technically compute the expectation value of any analytic, smooth observable via a Taylor expansion:
$$\langle\mathcal{O}(z)\rangle = \langle \sum_{M}\frac{1}{M!}\sum_{\mu_1\ldots\mu_M}^N \frac{\partial^m \mathcal{O}}{\partial z_{\mu_1}\ldots z_{\mu_M}}|_{z=0}z_{\mu_1}\ldots z_{\mu_M}\rangle$$
$$\therefore \langle\mathcal{O}(z)\rangle = \sum_{M}\frac{1}{M!}\sum_{\mu_1\ldots\mu_M}^N \frac{\partial^m \mathcal{O}}{\partial z_{\mu_1}\ldots z_{\mu_M}}|_{z=0}\,\langle z_{\mu_1}\ldots z_{\mu_M}\rangle$$
Doing this actual computation and average, though, quickly becomes unfeasible for high-order correlators (i.e. large $M$). In reality, we often describe PDFs with a smaller set of correlators/quantities which describe most of the useful stuff. 

For a Gaussian distribution, it is totally described by the kernel $K^{\mu\nu}$. All higher order correlators are combinations of that kernel tensor. 

We can ask a more tractable question: How non-Gaussian is our PDF? To get an answer, we use what statisticians call cumulants and what physicists call connected correlators/diagrams. You can also distinguish the connected correlator from the full correlator. Here are the definitions of the connected correlator. 
$$\langle z_\mu\rangle_{\textrm{connected}} = \langle z_\mu \rangle$$
$$\langle z_{\mu} z_\nu\rangle_{\textrm{connected}} = \langle z_\mu z_\nu \rangle - \langle z_\mu\rangle\langle z_\nu\rangle$$
$$\therefore \langle z_{\mu} z_\nu\rangle_{\textrm{connected}}= \langle \,(z_\mu - \langle z_\mu\rangle)\cdot(z_\nu - \langle z_\nu\rangle)\,\rangle \equiv \textrm{Covariance[$z_\mu$,$z_\nu$]}$$
The two-point connected correlator is the covariance. For Gaussian PDFs, it just amounts to the kernel $K_{\mu\nu}$. Notice that what we're doing is subtracting any contributions to the full two-point correlator which arise from the single-point correlator (i.e. the mean). 

Let's look at $M=4$. We'll assume that there's no three-point connected correlator. Let's remove all lower-order contributions. We get this pretty lengthy expression.
$$\langle z_{\mu_1}z_{\mu_2}z_{\mu_3}z_{\mu_4}\rangle|_{\textrm{conn.}} = \langle z_{\mu_1}z_{\mu_2}z_{\mu_3}z_{\mu_4}\rangle - \sum_{\textrm{all combos}}\langle z_{\mu_{k_1}}z_{\mu_{k_2}}\rangle|_{\textrm{conn.}}\langle z_{\mu_{k_3}}z_{\mu_{k_4}}\rangle|_{\textrm{conn.}} - \sum_{\textrm{all combos}}\langle z_{\mu_{k_1}}z_{\mu_{k_2}}\rangle|_{\textrm{conn.}}\langle z_{\mu_{k_3}}\rangle \langle z_{\mu_{k_4}}\rangle - \langle z_{\mu_1}\rangle \langle z_{\mu_2}\rangle \langle z_{\mu_3}\rangle \langle z_{\mu_4}\rangle$$
For a Gaussian distribution, this four-point connected correlator is zero. You can see this quickly for a zero-mean Gaussian, since this just becomes a sum of kernel tensor products; Naturally, it goes to zero. Thus, if you observed a non-zero value for the four point connected correlator, you know your PDF is different from a basic Gaussian. Note that if we assume a parity invariant distribution under $z\rightarrow -z$, we can set to zero all odd M-point correlators and all single-point means. 

We can now describe our general formula for a M-point connected correlator. 
	**Basically, we want to subtract any contributions to our M-point total correlator which arise from lower order connected correlators. Said another way, we want our M-point connected correlator to be an indicator of some special term that arises from a genuinely new contribution to the M-point correlator that isn't just a product of lower-order correlators.** 

$$\langle z_{\mu_1}\ldots z_{\mu_M} \rangle = \langle z_{\mu_1}\ldots z_{\mu_M}\rangle|_{\textrm{conn.}} + \sum_{\textrm{all subdivisions}}\langle z_{\mu_{k_1^{[1]}}}\ldots z_{\mu_{k_{\nu_1}^{[1]}}}\rangle|_{\textrm{conn.}}\ldots \langle z_{\mu_{k_{\nu_1}^{[s]}}}\ldots z_{\mu_{k_{\nu_s}^{[s]}}}\rangle|_{\textrm{conn.}} $$
This is a little confusing in terms of notation. We're fundamentally summing up over all possible lower-order connected correlators and their possible index orderings. So, the $[s]$ represents however many $m$-point connected correlators there are for $m < M$. Thus, if wrote out our full expression for the four-point connected correlator, we would subtract all two-point connected correlators and all single-point correlators, thus remove all lower-order contributions. 

**In terms of quantum field theory, this amounts to looking at the literal $M$-point vertex/interaction term in our Lagrangian. We don't want to probe contributions from lower-order vertices in our Lagrangian, just the actual $M$-point vertex/interaction term. That term is distinctly unique from its lower order terms. If you think about things visually in this way (based on QFT), it's pretty clear what connected correlators are. We're looking for the connected Feynman diagrams, basically.**

We can thus define a **nearly-Gaussian distribution** as one where all high order M-point connected correlators for $M>2$ are small compared to their lower-order connected correlators. **In QFT speak, a nearly-Gaussian distribution is one where the higher-order couplings or interaction terms can be treated perturbatively.**

### 1.3 Nearly-Gaussian Distributions
Let's now get things closer to field theory. We can describe PDFs via an action $S(z)$. 
$$p(z)\propto e^{-S(z)}$$
Naturally, this is normalised s.t. $\int d^N z \, p(z) = 1$. We can define a partition function or normalisation factor. 
$$Z = \int d^N z\,e^{-S(z)}$$
With this, we write our PDF as follows:
$$p(z) = \frac{e^{-S(z)}}{Z}$$
We can relate a PDF to an action via $S(z) = - \log p(z)$. 

#### Quadratic Action and Gaussian Distribution
Recall our basic Gaussian distribution. We can recast in terms of an action.
$$S(z) = \frac{1}{2}K^{\mu\nu}z_\mu z_\nu$$
$$Z = I_K = \sqrt{|2\pi K|}$$
This is our basic distribution. We will define expectation values about this PDF and notate them as follows:
$$\langle\mathcal{O(z)}\rangle_K = \frac{1}{\sqrt{|2\pi K|}}\int d^N z\,e^{-\frac{1}{2}K^{\mu\nu}z_\mu z_\nu}\,\mathcal{O(z)}$$
General expectation values for general PDFs will be denoted as just $\langle \mathcal{O}(z)\rangle$. 

#### Quartic Action and Perturbation Theory
Let's now introduce a higher-order term to our action. We'll do this perturbatively, similar to the classical $\phi^4$ theory we often study in classical and quantum field theory. 
$$S(z) = \frac{1}{2}K^{\mu\nu}z_\mu z_\nu + \frac{\epsilon}{4!}V^{\mu\nu\rho\lambda} z_\mu z_\nu z_\rho z_\lambda$$
Evidently, the fourth order term is of order $\mathcal{O}(\epsilon)$. Thus, it's a small perturbation to our quadratic, purely Gaussian action. Let us assume that this fourth-order coupling tensor is **totally symmetric about all its indices**.  Let's now compute the partition function. 
$$Z = \int d^N z \,\exp[\,-\frac{1}{2}K^{\mu\nu}z_\mu z_\nu - \frac{\epsilon}{4!}V^{\mu\nu\rho\lambda} z_\mu z_\nu z_\rho z_\lambda\,]$$
$$= I_K\,\frac{1}{I_K}\int d^N z \,\exp[\,-\frac{1}{2}K^{\mu\nu}z_\mu z_\nu\,]\exp[\, - \frac{\epsilon}{4!}V^{\mu\nu\rho\lambda} z_\mu z_\nu z_\rho z_\lambda\,]$$
$$\therefore Z =\sqrt{|2\pi K|} \,\langle\exp[\,- \frac{\epsilon}{4!}V^{\mu\nu\rho\lambda} z_\mu z_\nu z_\rho z_\lambda\,]\rangle_K$$
Doing this integral exactly is very difficult, but we can compute this perturbatively to $\mathcal{O}(\epsilon)$.
$$Z \approx \sqrt{|2\pi K|} \,\langle\,[\,1- \frac{\epsilon}{4!}V^{\mu\nu\rho\lambda} z_\mu z_\nu z_\rho z_\lambda + \mathcal{O}(\epsilon^2)\,]\,\rangle_K$$
$$\approx \sqrt{|2\pi K|} \,[\,1- \frac{\epsilon}{4!}V^{\mu\nu\rho\lambda} \langle\,z_\mu z_\nu z_\rho z_\lambda\,\rangle_K\,] + \mathcal{O}(\epsilon^2)$$
$$\approx \sqrt{|2\pi K|} \,[\,1- \frac{\epsilon}{4!}V^{\mu\nu\rho\lambda} (\,K_{\mu\nu}K_{\rho\lambda} + K_{\mu\rho}K_{\nu\lambda} + K_{\mu\lambda}K_{\rho\nu}\,)\,] + \mathcal{O}(\epsilon^2)$$
$$Z \approx \sqrt{|2\pi K|} \,(\,1- \frac{\epsilon}{8}V^{\mu\nu\rho\lambda} K_{\mu\nu}K_{\rho\lambda}\,) + \mathcal{O}(\epsilon^2)$$
where we've used the symmetry of the fourth-order coupling tensor (and relabelling the indices) to group together all the kernel products. Recall that we're using the Einstein summation convention. 

Let us now compute the two-point correlator. 
$$\langle z_{\mu_1}z_{\mu_2} \rangle = \frac{1}{Z}\int d^N z\, e^{-S(z)}\,z_{\mu_1}z_{\mu_2}$$
$$= \frac{\sqrt{|2\pi K|}}{Z} \langle\,z_{\mu_1}z_{\mu_2}\, \exp[\, - \frac{\epsilon}{4!}V^{\mu\nu\rho\lambda} z_\mu z_\nu z_\rho z_\lambda\,]\,\rangle_K$$
$$= \frac{\sqrt{|2\pi K|}}{Z}\,[\,\langle\,z_{\mu_1}z_{\mu_2}\,\rangle_K - \frac{\epsilon}{4!}V^{\mu\nu\rho\lambda} \langle\,z_\mu z_\nu z_\rho z_\lambda z_{\mu_1}z_{\mu_2}\,\rangle_K\,)\,] + \mathcal{O}(\epsilon^2)$$
$$= \frac{\sqrt{|2\pi K|}}{Z}\,[\,K_{\mu_1\mu_2} - \frac{\epsilon}{4!}V^{\mu\nu\rho\lambda} \langle\,z_\mu z_\nu z_\rho z_\lambda z_{\mu_1}z_{\mu_2}\,\rangle_K\,)\,] + \mathcal{O}(\epsilon^2)$$
$$= (\,1 + \frac{\epsilon}{8}V^{\mu\nu\rho\lambda} K_{\mu\nu}K_{\rho\lambda}\,)K_{\mu_1\mu_2}\,-\,\frac{\epsilon}{4!}V^{\mu\nu\rho\lambda} \langle\,z_\mu z_\nu z_\rho z_\lambda z_{\mu_1}z_{\mu_2}\,\rangle_K \,+\, \mathcal{O}(\epsilon^2)$$
$$= (\,1 + \frac{\epsilon}{8}V^{\mu\nu\rho\lambda} K_{\mu\nu}K_{\rho\lambda}\,)K_{\mu_1\mu_2}\,-\,\frac{\epsilon}{4!}V^{\mu\nu\rho\lambda} (\,3 K_{\mu_1\mu_2}K_{\mu\nu}K_{\rho\lambda} + 12 K_{\mu_1\mu}K_{\mu_2\nu}K_{\rho\lambda}\,) \,+\, \mathcal{O}(\epsilon^2)$$
$$\langle z_{\mu_1} z_{\mu_2}\rangle= K_{\mu_1\mu_2} - \frac{\epsilon}{2}V^{\mu\nu\rho\lambda}K_{\mu_1\mu}K_{\mu_2\nu}K_{\rho\lambda} + \mathcal{O}(\epsilon^2)$$
Note that in deriving the second last line, we can expand out the six-point correlator which should have 15 independent terms. If this isn't clear, recall that for a $2m$-point correlator, there are a total of $(2m)!$ different ways of arranging all the indices. We then divide by $(2!)^m$ because we don't care about the ordering of the indices in each kernel. Next, we divide by a factor of $m!$ because we don't care whether the indices appear in the first kernel or the last $m$-th kernel. Thus, we always get the following number independent terms in a correlator:
$$N_{\textrm{independent terms}}(2m-\textrm{pt. corr.}) = \frac{(2m)!}{m!\cdot (2!)^m}$$
Thus, we have 15 terms for our 6-pt correlator of which only three have $\mu_1\mu_2$ on the same kernel. The other 12 have them on different kernels. The terms which groups them on one kernel gets cancelled out with the term arising from Taylor expanding the denominator, leaving us with our simplified final expression. 

Thus, we've computed the $\mathcal{O}(\epsilon)$ correction to the two-point function for our quartic action. Let's compute the four-point correlator:
$$\langle z_{\mu_1}z_{\mu_2}z_{\mu_3}z_{\mu_4}\rangle = \frac{1}{Z}\int d^N z\, e^{-S(z)}\,z_{\mu_1}z_{\mu_2}z_{\mu_3}z_{\mu_4}$$
$$=\frac{\sqrt{|2\pi K|}}{Z}[\,\langle z_{\mu_1}z_{\mu_2}z_{\mu_3}z_{\mu_4}\rangle_K - \frac{\epsilon}{4!}V^{\mu\nu\rho\lambda} \langle\,z_\mu z_\nu z_\rho z_\lambda z_{\mu_1}z_{\mu_2}z_{\mu_3}z_{\mu_4}\,\rangle_K\,] + \mathcal{O}(\epsilon^2)$$
$$= (\,1 + \frac{\epsilon}{8}V^{\mu\nu\rho\lambda} K_{\mu\nu}K_{\rho\lambda}\,)\langle z_{\mu_1}z_{\mu_2}z_{\mu_3}z_{\mu_4}\rangle_K - \frac{\epsilon}{4!}V^{\mu\nu\rho\lambda} \langle\,z_\mu z_\nu z_\rho z_\lambda z_{\mu_1}z_{\mu_2}z_{\mu_3}z_{\mu_4}\,\rangle_K\, + \mathcal{O}(\epsilon^2)$$
$$= (\,1 + \frac{\epsilon}{8}V^{\mu\nu\rho\lambda} K_{\mu\nu}K_{\rho\lambda}\,)(K_{\mu_1\mu_2}K_{\mu_3\mu_4} + K_{\mu_1\mu_3}K_{\mu_2\mu_4} + K_{\mu_1\mu_4}K_{\mu_2\mu_3})$$
$$- \frac{\epsilon}{4!}V^{\mu\nu\rho\lambda}\times$$
$$(\,3 K_{\mu_1\mu_2}K_{\mu_3\mu_4}K_{\mu\nu}K_{\rho\lambda} + 12 K_{\mu_1\mu_2}K_{\mu_3\mu}K_{\mu_4\nu}K_{\rho\lambda} + 12 K_{\mu_3\mu_4}K_{\mu_1\mu}K_{\mu_2\nu}K_{\rho\lambda}$$
$$+\,3 K_{\mu_1\mu_3}K_{\mu_2\mu_4}K_{\mu\nu}K_{\rho\lambda} + 12 K_{\mu_1\mu_3}K_{\mu_2\mu}K_{\mu_4\nu}K_{\rho\lambda} + 12 K_{\mu_2\mu_4}K_{\mu_1\mu}K_{\mu_3\nu}K_{\rho\lambda}$$
$$+\,3 K_{\mu_1\mu_4}K_{\mu_2\mu_3}K_{\mu\nu}K_{\rho\lambda} + 12 K_{\mu_1\mu_4}K_{\mu_2\mu}K_{\mu_3\nu}K_{\rho\lambda} + 12 K_{\mu_2\mu_3}K_{\mu_1\mu}K_{\mu_4\nu}K_{\rho\lambda}$$
$$+\,24 K_{\mu_1\mu}K_{\mu_2\nu}K_{\mu_3\rho}K_{\mu_4\lambda}\,\,)\, + \mathcal{O}(\epsilon^2)$$
where, again, we use the symmetry properties of the four-point interaction tensor to simplify things (note that all prefactors add up to 105, as expected). It's a pretty gross expression, where we split things up into groupings of the four $\mu_i$ indices. We can see immediately that the terms with the prefactor $3$ cancel out the $\mathcal{O}(\epsilon)$ terms in the first line because the 3 multiplies the factor of $1/4!$ to give $1/8$, just like the first line. We end up with the following large expression. 

To simplify things, let's use the following fact:
$$\langle z_{\mu_1}z_{\mu_2}\rangle\cdot\langle z_{\mu_3}z_{\mu_4}\rangle = (\,K_{\mu_1\mu_2} - \frac{\epsilon}{2}V^{\mu\nu\rho\lambda}K_{\mu_1\mu}K_{\mu_2\nu}K_{\rho\lambda} + \mathcal{O}(\epsilon^2)\,)\cdot(1\rightarrow 3,\,2\rightarrow4)$$
$$\therefore \langle z_{\mu_1}z_{\mu_2}\rangle\cdot\langle z_{\mu_3}z_{\mu_4}\rangle = K_{\mu_1\mu_2}K_{\mu_3\mu_4} - \frac{\epsilon}{2}V^{\mu\nu\rho\lambda}(\,K_{\mu_3\mu_4}K_{\mu_1\mu}K_{\mu_2\nu}K_{\rho\lambda} + K_{\mu_1\mu_2}K_{\mu_3\mu}K_{\mu_4\nu}K_{\rho\lambda}\,) + \mathcal{O}(\epsilon^2)$$
This realisation allows us to re-express the four-point correlator from preceeeding lines above. 
$$\langle z_{\mu_1}z_{\mu_2}z_{\mu_3}z_{\mu_4}\rangle = \langle z_{\mu_1}z_{\mu_2}\rangle\langle z_{\mu_3}z_{\mu_4}\rangle + \langle z_{\mu_1}z_{\mu_3}\rangle\langle z_{\mu_2}z_{\mu_4}\rangle + \langle z_{\mu_1}z_{\mu_4}\rangle\langle z_{\mu_2}z_{\mu_3}\rangle $$
$$ - \epsilon \,V^{\mu\nu\rho\lambda}K_{\mu_1\mu}K_{\mu_2\nu}K_{\mu_3\rho}K_{\mu_4\lambda} + \mathcal{O}(\epsilon^2)$$
Thus, we see that the connected correlator is given by the following:
$$\langle z_{\mu_1}z_{\mu_2}z_{\mu_3}z_{\mu_4}\rangle|_{\textrm{connected}} =  - \epsilon \,V^{\mu\nu\rho\lambda}K_{\mu_1\mu}K_{\mu_2\nu}K_{\mu_3\rho}K_{\mu_4\lambda} + \mathcal{O}(\epsilon^2)$$
Thus, the connected correlator brings out this quartic coupling term and eliminates all the other disconnected correlators. In QFT-speak, we only care about the Feynman diagram that arises due to this quartic coupling term. This is a clear non-Gaussian property. This introduces non-trivial fourth-order correlations between inputs $z$ which also shift the two-point correlator, not just the four-point correlator. 

**We will show later on in this textbook that this perturbative expansion in the small parameter $\epsilon$ is deeply related to expanding neural networks in $1/n$ where $n$ is the network width.** 

**THIS SECTION IS COOL, BUT WOULD BE GOOD TO HAVE SOME MORE PHYSICAL INTUITION, CONNECT THINGS DIRECTLY TO QFT, PROPAGATORS, LOOP DIAGRAMS, ETC. THAT WOULD ADD A LOT TO THIS!**

#### Statistical independent and interactions
In QFT, the quartic action is one of the simplest types of interacting field theories. This relates deeply to the notion of statistical independence. 

For the joint PDF of two random variables $x,y$ are independent, it has the following form:
$$p(x,y) = p(x)p(y)$$
Now, imagine our Gaussian action. Statistically, if we pick the diagonal basis of the kernel tensor $K_{\mu\nu}$, all the components are independent. We pick the basis where $(O\cdot K\cdot O^T)_{\mu\nu}=\lambda_\mu \delta_{\mu\nu}$, and $(O\cdot z)_\mu = u_\mu$. 
$$p(z) = \frac{1}{\sqrt{|2\pi K|}}\exp\Big[\,-\sum_\mu^N \frac{u_\mu^2}{2\lambda_\mu}\,\Big] = \Pi_{\mu=1}^N\bigg(\,\frac{e^{-u_\mu^2/2\lambda_\mu}}{\sqrt{2\pi \lambda_\mu}}\,\bigg)\equiv p(u_1)\ldots p(u_N)$$
Thus, we see that the Gaussian theory (or, in QFT-speak, the free-field theory) is essentially a totally statistically independent theory. All variables are decoupled, all correlations are lost. 

This kind of decoupling between variables (i.e. statistical independence) is very difficult to do for non-Gaussian terms in the action. Thus, generally, having non-Gaussian terms in your action will break down the statistical independence of these variables. **Non-gaussian interaction terms result in the breakdown of statistical independence and the introduction of high-order correlations between variables.** 

**THINK OF THE PHYSICAL DESCRIPTION OF THIS. NEED BETTER PHYSICAL ANALOGIES CONNECTED TO QFT, TO TREE-LEVEL DIAGRAMS, TO LOOP DIAGRAMS, ETC.**

#### Thoughts on General Nearly-Gaussian Actions
Let's write down a general non-Gaussian action where we assume parity symmetry $z\rightarrow -z$, so no odd terms can exist in the action:
$$S(z) = \frac{1}{2}K^{\mu\nu}z_\mu z_\nu + \sum_{m=2}^k\frac{1}{(2m)!}\,s^{\mu_1\ldots\mu_{2m}}z_{\mu_1}\ldots z_{\mu_{2m}}$$
The coefficients $s^{\mu_1\ldots \mu_{2m}}$ are the non-Gausian couplings which control the interactions between $z_\mu$ components. The presence of these order $2m$ coupling terms directly controls the ability of the model to demonstrate non-Gaussian correlations between different $\mu$ components. They are directly probed by the connected correlators of degree $2m$. 

*A note on dimensions: If we give dimensions $[z_\mu] = \xi$, then $[K^{\mu\nu}]=\xi^{-2}$ and the inverse matrix has dimensions $[K_{\mu\nu}] = \xi^{2}$. Higher order coupling terms have dimension $[s^{\mu_1\ldots \mu_{2m}}]=\xi^{-2m}$.* 

What do we mean by nearly-Gaussian distributions? We mean that higher-order terms in the action have the following perturbative property:
$$|s^{\mu_1\ldots \mu_{2m}}| \ll |K^{\mu\nu}|^m$$
This, naturally, is an abuse of notation but the dimensions and idea are right. We want these perturbative terms to be small compared to their Gaussian counterparts. Said another way, **we want a perturbative, nearly-Gaussian action to have its connected diagrams/correlators subdominant to the disconnected diagrams/correlators.**

We will find that wide neural networks are described by nearly-Gaussian distributions where the connected correlators are hierarchically small: $$\langle z_{\mu_1}\ldots z_{\mu_{2m}}\rangle|_{\textrm{connected}} \sim \mathcal{O}(\epsilon^{m-1})$$
We already saw this earlier with our quartic action for $m=2$ being of order $\mathcal{O}(\epsilon)$. We will demonstrate this behaviour for neural networks in the infinite width (i.e. large $n$) limit. It will be sufficient to use a quartic action and keep terms up to order $\mathcal{O}(\epsilon^2)$. 


**IDEA: IF IN THE INFINITE-WIDTH LIMIT WE FIND THAT NON-GAUSSIAN CORRELATIONS VANISH, THEN THAT MEANS THAT THREE-POINT COVARIANCE AND OTHER HIGH-ORDER CORRELATIONS SHOULD VANISH. WE SHOULD BE ABLE TO SHOW THIS WITH DATA OF THE FORM $f(x_i) = x_1 x_2$, $f(x_i) = x_1 x_2 x_3$ AND $f(x_i) = x_1 x_2 x_3 x_4$ FOR VERY HIGH-DIMENSIONS $N_i > 10$. WE SHOULD BE ABLE TO SHOW THAT AS WIDTH INCREASES, THE CORRELATIONS GO AWAY FOR OUR TOY-DATASETS... CAN ALSO MODEL THE LARGE-WIDTH LIMIT WITH THE NEURAL FEATURE ANSATZ THAT MISHA BELKIN DEVELOPED.** 


## Chapter 2: Neural Networks
This chapter defines some fundamental mathematical notation for fully connected networks (FCNs) that forms the basis for the rest of the textbook. Note that I prefer FCN to MLP because these neural networks (NNs) that we'll discuss are literally fully connected networks (FCN). Multi-Layer perceptron (MLP) does make sense historically but it's not very intuitive to me from a mathematical point of view. Whatever...

### 2.1 Function Approximation
Suppose there's a true function that we want to approximate. This true function is labelled $f(x)$.

The fundamental idea behind all of ML is function approximation. Basically, from a large set of extremely flexible functions $\{f(x;\theta)\}$ characterised by some parameters $\theta$. We then tune the parameters $\theta$ to achieve some $\theta^*$ s.t. $f(x;\theta^*)\approx f(x)$. This is function approximation. How we update the parameters is called the learning algorithm. 

Let us represent our dataset as $\mathcal{D}$ where each input data vector has dimension $n_0$: 
$$\mathcal{D} = \{x_{i;\alpha}\}_{\alpha=1,\ldots,N_D}$$
where $\alpha$ is the individual data index and there are $N_D$ datapoints. $i$ is the input data vector dimension index, so $i\in \{1,\ldots,n_0\}$. 

Let's now describe each component of our flexible FCN architecture. We'll look at each individual neuron:
- **Preactivation**: $z_i(s) = W_{ij}s_j + b_i$ which goes from $s_j\in \mathbb{R}^{n_{in}}$ and $i\in \{1,\ldots,n_{out}\}$. This preactivation takes an input vector $s_j$ in $n_{in}$-dimensional input space to $n_{out}$-dimensional output space. 
- **Activation**: Now, each component of the preactivation is passed through an activation function: $\sigma_i = \sigma(z_i)$ where $\sigma(x)$ acts on each component of the preactivation $z_i$. This is a non-linear function which allows the FCN to express non-linear functions of the input variables. Without this, FCNs wouldn't be particularly interesting (they'd just be large affine, linear functions...).
These $n_{out}$ neurons form a layer. We often call $b_i$ our biases and $W_{ij}$ our weights. By chaining together many of these preactivations and activations across several layers, we get our FCNs (aka MLPs). This is the total FCN functional form:

$$z_i^{(1)}(x_\alpha) = W_{ij}^{(1)} x_{j;\alpha} + b_i^{(1)}$$ for $i = 1,\ldots,n_1$ and $j=1,\ldots,n_0$.  
$$z_i^{(l)}(x_\alpha) = W_{ij}^{(l)} \sigma(z_{j}^{(l-1)}(x_\alpha)) + b_i^{(l)}$$
for $i = 1,\ldots,n_{l}$ and $l = 2,\ldots,L$.

where we're using Einstein summation notation. Also note that $W_{ij}^{(l)}$ isn't necessarily a square matrix. Specifically, $n_{l-1}$ is not necessarily the same as $n_{l}$. $L$ is the depth of the FCN and $n_1,\ldots, n_{L}$ are the widths of each layer of the FCN. The depth, widths, and activation functions are examples of hyperparameters that we can choose. If we want a one-layer network $L=1$, then we don't have to worry about activation functions: We just have an affine linear function of the input data vector. For $L=2$, we have that initial affine linear layer, which gets acted on by an activation function and fed to another affine linear final layer. 

With all of this, we end up with the following function approximation:
$$f(x;\theta) = z^{(L)}(x)$$
The total number of parameters $\theta$ in the FCN is giving by the following:
$$N_\theta = \sum_{l=1}^L (n_l\cdot n_{l-1} + n_l)$$
where the floating $n_l$ comes from the biases. We can see that the number of parameters scale quadratically with the widths and linearly with depth (at the twiddle level, the basic scaling):
$$N_{\theta}\propto n^2 L$$
The choice of NN architecture basically just amounts to choosing inductive biases about the form of these flexible functions and network parameters. Essentially, they amount to constraints on the relationship between weights. 
- For computer vision, convolutional neural networks (CNNs) are common and use translation equivariance in their functional forms.
- Natural language processing (NLP) often uses transformers (think ChatGPT, etc.). These architectures encourage long-range correlations between language elements in embedding spaces, a property known as attention. 

Much of the calculation in this textbook can be done for different architectures by simply replacing our FCN iteration equations with those of the other architectures. We may show these functional forms explicitly for CNNs, transformers, ResNets, etc. in the appendices but we'll focus on FCNs for all the main calculations. 

Before moving on, I want to point out that there's already a large degeneracy in the parameter space with large $n$. Specifically, if we consider the weights in a given layer, $W_{ij}^{(l)}$, multiplied by the weights for the previous layer $W_{jk}^{(l-1)}$, notice that there is a dummy/summed index $j$. Thus, we can permute the $j$ indices however we like and we get the exact same result. You can permute those indices $n_l!$ different ways. Note that this doesn't care about whether there's an activation function acting on one of the weight matrices (as there typically will be), because you still sum over those indices. As you chain together more layers, you continually multiply by $n_{l-1}!$ for every $l$. That is an enormous degeneracy!
$$\textrm{degeneracy in $\theta$} = \Pi_{l=1}^{L-1}\,(n_l!)$$

### 2.2 Activation Functions
#### Perceptron
$$
\sigma(z)=
\begin{cases}
1, & z \ge 0,\\
0, & z < 0 .
\end{cases}
$$
It either fires or not. It's either 1 or 0. Historically significant but not very used nowadays.

#### Sigmoid
$$\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{1}{2}\Bigg(1+\tanh\Big(\frac{z}{2}\Big)\Bigg)$$
This is a smooth, differentiable function. It has the limits $\sigma\rightarrow 0$ for $z\rightarrow -\infty$ and $\sigma\rightarrow 1$ for $z\rightarrow \infty$. At the origin, $\sigma(0) = 1/2$. Not very used nowadays because it doesn't cross the origin at 0.

#### Tanh
$$\sigma(z) = \tanh(z) = \frac{e^z - z^{-z}}{e^z + e^{-z}} = \frac{e^{2z} - 1}{e^{2z}+1}$$
Also, nice, smooth, and differentiable. As $z\rightarrow -\infty$, $\sigma(z)\rightarrow -1$ and as $z\rightarrow \infty$, $\sigma(z)\rightarrow 1$. At $z=0$, it crosses the origin $\sigma(0) = 0$. It is a very popular choice nowadays aside from the ReLU. 

#### Sin
$$\sigma(z) = \sin(z)$$
Cool for periodic functions, but not super popular nowadays. 

#### Scale-Invariant: Linear, ReLU, and leaky ReLU
This is any activation function that has the form:
$$\sigma(\lambda z) = \lambda \sigma(z)$$
To achieve this, we use the following form:
$$
\sigma(z)=
\begin{cases}
a_+ z, & z \geq 0,\\
a_- z, & z < 0 .
\end{cases}
$$
- Linear function: $a_-=a_+=1$
- ReLU: $a_- = 0$, $a_+ = 1$
- Leaky ReLU: $a_-=a$ for some small $a < 1$. $a_+ = 1$. This is like ReLU but allows some small, non-zero slope for negative $z$. 

ReLU is by far the most popular activation function. It will form the basis of the theory in this book but, as always, you can rederive a bunch of this theory for other choices of activation function. 

Now, why is this called scale-invariant? Fundamentally, a scale-invariant function is one for which zooming in and out of the input variable space doesn't change the shape of the function at all. Think of the basic linear function $\sigma(z) = z$. You can scale the input space $z\rightarrow \lambda z$ which will amount to shrinking or stretching the input axis depending on the value of $\lambda$, but $\sigma(\lambda z) = \lambda z$. It looks exactly the same in the stretched/shrunken input space (just replace $\lambda z\equiv x$ and you get $\sigma(x) = x$, boring). Same thing with ReLU, leaky ReLU. Note that if we translate the function, it no longer has this nice property. But we only care about scaling the input space, not translating it. 

The other activation functions we've looked at are not scale-invariant. The reach a certain threshold $|z|$ value and the information saturates. This doesn't happen for ReLU, leaky ReLU, or the linear function. 

Final note: Linear activations are boring. We want non-linear activations, not linear ones. 

#### ReLU-like: softplus, SWISH, GELU
These arise to make ReLU differentiable and smooth at the origin. 

- **Softplus**: 
$$\sigma(z) = \log(1 + e^z)$$
This is is 0 in the limit $z\rightarrow -\infty$, it is $\log(2)$ at $z=0$, and it's linear $\sigma(z)\approx z$ in the limit $z\rightarrow \infty$.
 
- **SWISH**: 
$$\sigma(z) = \frac{z}{1 + e^{-z}}$$
This is 0 at the origin, becomes linear $\sigma(z)\approx z$ as $z\rightarrow \infty$, and approaches 0 in the limit $z\rightarrow -\infty$. Note that it does go negative for $z<0$ but when $|z|$ is not large.

- **GELU**: (Gaussian Error Linear Unit)
$$\sigma(z) = \Bigg[\frac{1}{2} + \frac{1}{2}\textrm{erf}\bigg(\frac{z}{\sqrt{2}}\bigg)\Bigg]$$
where $$\textrm{erf}(z) = \frac{2}{\sqrt{\pi}}\int_0^z dt\,e^{-t^2}$$
is the Gaussian error function, a partial integration of a unit-Gaussian distribution (the factor of 2 is because it's symmetric about $-z$ and $z$). This is very similar to the SWISH activation function. 

Note that all these three choices, in smoothing the ReLU, break scale-invariance. 
![[Pasted image 20251102231257.png]]

### 2.3 Ensembles
Before training FCNs, you need to pick some initial parameters. How you initialise can influence how well training goes. You could just pick 0 for all weights and biases. However, that makes all neurons in each hidden layer exactly identical. That makes updating the weights extremely uninteresting because each neuron in a layer are updated identically. 

A better approach to initialisation is just to pick from some PDF of the weights and biases. Ideally, we pick this initialisation distribution s.t. the resulting ensemble of trained networks are well-behaved w.r.t our function approx. task. 

#### Initialisation distribution of biases and weights
Let's use a Gaussian initialisation distribution with the following variances:
$$\langle b_{i_1}^{(l)} b_{i_2}^{(l)}\rangle = \delta_{i_1 i_2}C_b^{(l)}$$
$$\langle W_{i_1 j_1}^{(l)} W_{i_2 j_2}^{(l)}\rangle = \delta_{i_1 i_2}\delta_{j_1 j_2} \frac{C_W^{(l)}}{n_{l-1}}$$
This gives us the following PDFs:
$$p\Big(b_i^{(l)}\Big) = \frac{1}{\sqrt{2\pi C_b^{(l)}}}\,\exp\bigg[\,-\frac{1}{C_b^{(l)}}(\,b_i^{(l)}\,)^2\,\bigg]$$
$$p\Big(W_{ij}^{(l)}\Big) = \sqrt{\frac{n_{l-1}}{2\pi C_W^{(l)}}}\,\exp\bigg[\,-\frac{n_{l-1}}{C_W^{(l)}}\Big(\,W_{ij}^{(l)}\,\Big)^2\,\bigg]$$
Note that the factor of $1/n_{l-1}$ is important when actually computing the initialisation variances of FCNs in the upcoming chapter. To ruin the punchline, if you ignore the biases and just compute the expectation value $\langle z^{(l)}_{\alpha}\cdot z^{(l)}_\alpha\rangle \sim \langle W^{(l)}\cdot W^{(l)}\rangle \,\sigma^{(l-1)}\cdot \sigma^{(l-1)} \sim \Big(C_W^{(l)}/{n_{l-1}}\Big)\cdot n_{l-1}$. The factors cleanly cancel each other out. That's the reason for the convention. 

This set of layer-by-layer variances are known as initialisation hyperparameters: $\{C_b^{(1)},\ldots,C_b^{(L)}\}$ and $\{C_W^{(1)},\ldots,C_W^{(L)}\}$. We will show prescriptions for setting these initialisation hyperparams s.t. the outputs of our initialised FCNs are well behaved. 

#### Induced distributions
By generating a random distribution of initialised parameters, one also induces a random distribution of initialised NN outputs. This is before any training procedure. Recall what we mean by NN output:
$$f_i(x_\alpha;\theta) = z^{(L)}_i(x_\alpha)$$
where $\alpha$ are the data sample indices for dataset $\mathcal{D}$, and $i$ are the output dimension indices. Having control of this initial distribution $p(z^{(L)}|\mathcal{D})$ is of significant interest, since it directly controls how easy it is to train NNs. In theory, to compute this distribution, we need to compute a gigantic integral over initialised parameter space:
$$p(z^{(L)}|\mathcal{D}) = \int \Big[\Pi_{\mu=1}^{N_\theta}d\theta_\mu\Big]\,p(z^{(L)}|\theta,\mathcal{D})\,p(\theta)$$
where we've written all the $N_\theta$ parameters of the NN into a big array $\theta_\mu$. We know $p(\theta)$ because that is our init hyperparameter choice. We can also calculate $p(z^{(L)}|\theta,\mathcal{D})$ analytically. It is deterministic, actually, because given a specific set of parameters $\theta_\mu$, then there is exactly one possible output $z^{(L)}$. It's a delta function, basically, $\delta(z_i - z_i^{(L)}(x_\alpha|\theta_\mu))$. With this, let's compute the induced output distribution at initialisation.

#### Induced distributions, redux
Start with a one-layer FCN, $L=1$. 
$$p(z^{(1)}|\mathcal{D}) = \int \Big[\Pi_{i=1}^{n_1}db_i^{(1)}\,p\Big(b_i^{(1)}\Big)\Big]\,\Big[\Pi_{i=1}^{n_1} \Pi_{j=1}^{n_0}dW_{ij}^{(1)}\,p\Big(W_{ij}^{(1)}\Big)\Big]\,$$
$$\times \bigg[\Pi_{i=1}^{n_1}\Pi_\alpha\,\delta\Big(z_{i;\alpha}^{(1)} - W_{ij}^{(1)}x_{j;\alpha} - b_i^{(1)}\,\Big)\bigg]$$

This is for a 1-layer NN. We can also compute this layer $l$ given a preceeding layer $l-1$. 
$$p(z^{(l)}|z^{(l-1)},\mathcal{D}) = \int \Big[\Pi_{i=1}^{n_l}db_i^{(l)}\,p\Big(b_i^{(l)}\Big)\Big]\,\Big[\Pi_{i=1}^{n_l} \Pi_{j=1}^{n_{l-1}}dW_{ij}^{(l)}\,p\Big(W_{ij}^{(l)}\Big)\Big]\,$$
$$\times \bigg[\Pi_{i=1}^{n_{l}}\Pi_\alpha\,\delta\Big(z_{i;\alpha}^{(l)} - W_{ij}^{(l)}\sigma(z_{j;\alpha}^{(l-1)}) - b_i^{(l)}\,\Big)\bigg]$$

This is very similar in form to the upper equation, but taking into account the intermediate activation functions. With these two basic building blocks, we will in the next chapter compute the full distribution for a general neural network $f_i(x_\alpha;\theta)$:

$$p(z^{(L)}|\mathcal{D}) = \int \Big[\Pi_{\mu=1}^{N_\theta}d\theta_\mu\Big]\,p(\theta)\,\Big[\,\Pi_{i=1}^{n_{L}}\Pi_{\alpha}\,\delta\Big(z_{i;\alpha}^{(L)}-f_i(x_\alpha;\theta)\Big)\,\Big]$$
We will tackle this integral for FCNs in Chapter 4. First, in the next chapter, we will warm up by computing this distribution for deep linear networks.

## Chapter 3: Effective Theory of Deep Linear Networks at Initialisation
This is the final warm-up chapter. We will just compute the output distribution for deep linear networks, which will introduce a bunch of the techniques and formalism we need before diving into general, nonlinear FCNs. 

### 3.1 Deep Linear Networks (DLNs)
Here's our basic form:
$$z_{i;\alpha}^{(l)} = W_{ij}^{(l)}z_{j;\alpha}^{(l-1)}+b_i^{(l)}$$
This is like taking our previous FCN architecture and using a linear activation function (so, pretty boring). Let's simplify things by setting our biases to zero (or, equivalently, by absorbing them into our definition of $z_{j;\alpha}^{l}$ by adding a $1$ to the very end of the vector). 
$$z_{i;\alpha}^{(l)} = W_{i j_{l-1}}^{(l)} W_{j_{l-1} j_{l-2}}^{(l-1)}\ldots W_{j_1 j_0}^{(1)}x_{j_0;\alpha}$$
Note that the produce of all these matrices are are equivalent to just one large matrix. Let us also use the following initialisation distributions.
$$\langle W_{i j}^{(l)}\rangle = 0$$
$$\langle W_{i_1 j_1}^{(l)}W_{i_2 j_2}^{(l)}\rangle = \delta_{i_1 i_2}\delta_{j_1 j_2}\frac{C_W^{(l)}}{n_{l-1}}$$
Let's consider the following simple observable:
$$\langle z_{i;\alpha}^{(l)}\rangle = \langle W_{i j_{l-1}}^{(l)} W_{j_{l-1} j_{l-2}}^{(l-1)}\ldots W_{j_1 j_0}^{(1)}x_{j_0;\alpha} \rangle = \langle W_{i j_{l-1}}^{(l)}\rangle\langle W_{j_{l-1} j_{l-2}}^{(l-1)}\rangle\ldots \langle W_{j_{1} j_{0}}^{(1)}\rangle x_{j_0;\alpha} = 0$$
Pretty simple, given that all weight matrices are totally independent and they have zero mean. Thus, the mean output for a deep linear net should be 0.

**TASK: CREATE A NOTEBOOK WHERE YOU DEFINITE SOME DEEP LINEAR NETWORK FUNCTIONS, SPECIFY THEIR ACTIVATION DISTRIBUTIONS AND HYPERPARAMS (WIDTH, DEPTH), AND DEMONSTRATE THAT THE MEAN OUTPUT OVER AN ENSEMBLE OF NETWORKS IS ZERO. THIS INCLUDES INTERMEDIATE LAYERS**
### 3.2 Criticality
Let's now look at the two-point correlator. It will encode the math needed to 
#### Recursion for the two-point correlator
Consider the first layer. 
$$\langle z_{i_1;\alpha_1}^{(1)}z_{i_2;\alpha_2}^{(1)}\rangle = \langle W_{i_1 j_1}^{(1)}W_{i_2 j_2}^{(1)}\rangle\,x_{j_1;\alpha_1}x_{j_2;\alpha_2} = \frac{C_W^{(1)}}{n_0}\delta_{i_1 i_2}\delta_{j_1 j_2}x_{j_1;\alpha_1}x_{j_2;\alpha_2} = \frac{C_W^{(1)}}{n_0}\delta_{i_1 i_2}x_{j;\alpha_1}x_{j;\alpha_2}$$
We can define the following function to simplify things:
$$G_{\alpha_1 \alpha_2}^{(0)} = \frac{1}{n_0}x_{j;\alpha_1}x_{j;\alpha_2}$$
where we are computing an Einstein summation. Thus:
$$\langle z_{i_1;\alpha_1}^{(1)}z_{i_2;\alpha_2}^{(1)}\rangle =\delta_{i_1 i_2}{C_W^{(1)}}G^{(0)}_{\alpha_1\alpha_2}$$
Let's do this same computation for an intermediate layer:
$$\langle z_{i_1;\alpha_1}^{(l)}z_{i_2;\alpha_2}^{(l)}\rangle = \langle W_{i_1 j_1}^{(l)}z_{j_1;\alpha_1}^{(l-1)}\,W_{i_2 j_2}^{(l)} z_{j_2;\alpha_2}^{(l-1)}\rangle = \delta_{i_1 i_2}C_W^{(l)}\frac{1}{n_{l-1}}\langle z_{j;\alpha_1}^{(l-1)}z_{j;\alpha_2}^{(l-1)}\rangle$$
where we've skipped some steps involving delta functions and simplified. Let us now define the following general function:
$$G_{\alpha_1\alpha_2}^{(l)} = \frac{1}{n_l}\langle z_{j;\alpha_1}^{(l)}z_{j;\alpha_2}^{(l)} \rangle$$
Thus, we can simplify our iterative function:
$$\langle z_{i_1;\alpha_1}^{(l)}z_{i_2;\alpha_2}^{(l)}\rangle = \delta_{i_1 i_2}C_W^{(l)}\,G_{\alpha_1\alpha_2}^{(l-1)}$$
Evidently, we can thus relate the general correlation functions over layers as follows:
$$\frac{1}{n_l}\langle z_{i;\alpha_1}^{(l)}z_{i;\alpha_2}^{(l)} \rangle = G_{\alpha_1\alpha_2}^{(l)} = \frac{n_l}{n_l}C_W^{(l)}\,G_{\alpha_1\alpha_2}^{(l-1)} = C_W^{(l)}\,C_W^{(l-1)}\,G_{\alpha_1\alpha_2}^{(l-2)} = \Big(\Pi_{i=1}^{l}C_W^{(i)}\Big)G_{\alpha_1\alpha_2}^{(0)}$$
where the $n_l$ in the numerator comes from summing up over all the neurons in layer $l$. Thus, we have the following nice recursive relation:
$$G_{\alpha_1\alpha_2}^{(l)} = \Big(\Pi_{i=1}^{l}C_W^{(i)}\Big)G_{\alpha_1\alpha_2}^{(0)}$$
Assuming all the layer variances are identical, we get the following clean relation.
$$G_{\alpha_1\alpha_2}^{(l)} = \Big(C_W\Big)^l\,G_{\alpha_1\alpha_2}^{(0)}$$

**CODE TASK: VERIFY THIS RECURSION RELATION EXPLICITLY WITH CODE!!!**

#### Physics: criticality
We can quickly notice two things:
- If $C_W > 1$, the covariance $G_{\alpha_1\alpha_2}^L$ blows up as $L\rightarrow \infty$. Thus, very deep FCNs will blow up in this limit and have very wide initial variances in their outputs.
- If $C_W < 1$, the covariance $G_{\alpha_1\alpha_2}^L$ vanishes as $L\rightarrow \infty$. Thus, very deep FCNs will collapse in this limit with vanishing small initial variances in their outputs. 

These limits are bad. They are the toy-models of the often observed vanishing and exploding gradients issue in FCNs. Either we have a numerically unstable initial condition or we have extreme information loss in the initial condition. Both are bad starting points for these networks. 

The only reasonable choice for very deep FCNs is $C_W = 1$. In this limit, we preserve the covariance across layers, allowing our initial network output to be stable and non-vanishing. This is our non-trivial fixed point. This is thus very much based on the physics of criticality. 

**CODE TASK: VERIFY THESE DIFFERENT REGIMES WITH CODE! VERY COOL. YOU CAN VARY OVER DEPTH $L$ and COVARIANCE $C_W$**.

### 3.3 Fluctuations
We now want to compute higher order correlators. We want to understand if the distribution of DLNs form a Gaussian distribution or something different. 

For this section, let's simplify our correlator functions by assuming our data is only a single input $x_\alpha = x$. We can introduce full dataset notation later and generalise things. 
$$G_2^{(l)}\equiv G_{\alpha\alpha}^{(l)} = G^{(l)}(x,x)$$
#### Recursion for the four-point correlator
We'll begin with the first layer.
$$\langle z_{i_1}^{(1)}z_{i_2}^{(1)}z_{i_3}^{(1)}z_{i_4}^{(1)}\rangle = \langle W_{i_1 j_1}^{(1)}W_{i_2 j_2}^{(1)}W_{i_3 j_3}^{(1)}W_{i_4 j_4}^{(1)}\rangle \,x_{j_1}x_{j_2}x_{j_3}x_{j_4}$$
$$ = \Bigg(\frac{C_W^{(1)}}{n_0}\Bigg)^2(\delta_{i_1 i_2}\delta_{j_1 j_2}\delta_{i_3 i_4}\delta_{j_3 j_4} + \delta_{i_1 i_3}\delta_{j_1 j_3}\delta_{i_2 i_4}\delta_{j_2 j_4} + \delta_{i_1 i_4}\delta_{j_1 j_4}\delta_{i_2 i_3}\delta_{j_2 j_3})\,x_{j_1}x_{j_2}x_{j_3}x_{j_4}$$
$$\langle z_{i_1}^{(1)}z_{i_2}^{(1)}z_{i_3}^{(1)}z_{i_4}^{(1)}\rangle = \Big(C_W^{(1)}\Big)^2(\delta_{i_1 i_2}\delta_{i_3 i_4} + \delta_{i_1 i_3}\delta_{i_2 i_4} + \delta_{i_1 i_4}\delta_{i_2 i_3})\,\Big(G_2^{(0)}\Big)^2$$
This is exactly the result we expect for a Gaussian distribution. This makes total sense because the first layer of a deep linear network is Gaussian. Will this hold for deeper layers? Before that, let's define some definitions for four-point correlators. Let's define the following:
$$\langle z_{i_1}^{(l)}z_{i_2}^{(l)}z_{i_3}^{(l)}z_{i_4}^{(l)}\rangle = (\delta_{i_1 i_2}\delta_{i_3 i_4} + \delta_{i_1 i_3}\delta_{i_2 i_4} + \delta_{i_1 i_4}\delta_{i_2 i_3})\,G_4^{(l)}$$
For the first layer, we get the following result:
$$G_4^{(1)} = \Big(C_W^{(1)}\Big)^2\Big(G_2^{(0)}\Big)^2$$
Let's now get the layer-by-layer iteration equation:
$$\langle z_{i_1}^{(l)}z_{i_2}^{(l)}z_{i_3}^{(l)}z_{i_4}^{(l)}\rangle = \langle W_{i_1 j_1}^{(l)}W_{i_2 j_2}^{(l)}W_{i_3 j_3}^{(l)}W_{i_4 j_4}^{(l)}\rangle \,\langle z_{j_1}^{(l-1)}z_{j_2}^{(l-1)}z_{j_3}^{(l-1)}z_{j_4}^{(l-1)}\rangle$$
$$ = \Bigg(\frac{C_W^{(l)}}{n_{l-1}}\Bigg)^2(\delta_{i_1 i_2}\delta_{j_1 j_2}\delta_{i_3 i_4}\delta_{j_3 j_4} + \delta_{i_1 i_3}\delta_{j_1 j_3}\delta_{i_2 i_4}\delta_{j_2 j_4} + \delta_{i_1 i_4}\delta_{j_1 j_4}\delta_{i_2 i_3}\delta_{j_2 j_3})\,\langle z_{j_1}^{(l-1)}z_{j_2}^{(l-1)}z_{j_3}^{(l-1)}z_{j_4}^{(l-1)}\rangle$$
$$=\Big(C_W^{(l)}\Big)^2(\delta_{i_1 i_2}\delta_{i_3 i_4} + \delta_{i_1 i_3}\delta_{i_2 i_4} + \delta_{i_1 i_4}\delta_{i_2 i_3})\,\frac{1}{n_{l-1}^2}\,\langle z_{j}^{(l-1)}z_{j}^{(l-1)}z_{k}^{(l-1)}z_{k}^{(l-1)}\rangle$$
Note the following way of rewriting the final part of this last line based on our previous definition of $G_4^{(l)}$:
$$\frac{1}{n_{l}^2}\,\langle z_{j}^{(l)}z_{j}^{(l)}z_{k}^{(l)}z_{k}^{(l)}\rangle = \frac{1}{n_l^2}(\delta_{jj}\delta_{kk} + \delta_{jk}\delta_{jk} + \delta_{jk}\delta_{jk})\,G_4^{(l)} = \Big(1 + \frac{2}{n_l}\Big)\,G_4^{(l)}$$
With this, we derive the following iterative equation:
$$G_4^{(l)} = \Big(C_W^{(l)}\Big)^2\Big(1 + \frac{2}{n_{l-1}}\Big)\,G_4^{(l-1)}$$
Thus, we get the following final formula:
$$G_4^{(l)} = \Big(C_W^{(l)}\Big)^2\Bigg[\,\Pi_{l'=1}^{l-1}\Big(C_W^{(l')}\Big)^2\bigg(1 + \frac{2}{n_{l'}}\bigg)\,\Bigg]\,\Big(G_2^{(0)}\Big)^2$$
Since we know that two-point function relation $G_{\alpha_1\alpha_2}^{(l)} = \Big(\Pi_{i=1}^{l}C_W^{(i)}\Big)G_{\alpha_1\alpha_2}^{(0)}$ holds, we can substitute that and get the following clean equation:
$$G_4^{(l)} = \Bigg[\,\Pi_{l'=1}^{l-1}\bigg(1 + \frac{2}{n_{l'}}\bigg)\,\Bigg]\,\Big(G_2^{(l)}\Big)^2$$
Let's now discuss the implications of this formula. Note that the factor in square brackets takes this four-point function away from the typical expectation from Gaussian probability distributions.
#### Physics: Large-$n$ expansion, non-Gaussianities, interactions, and fluctuations
For the four-point function, it simplifies significantly in the limit of infinite width where $n_l\rightarrow \infty$. 
$$G_4^{(l)} \rightarrow \Big(G_2^{(l)}\Big)^2$$
The full four-point correlator becomes the following:
$$\langle z_{i_1}^{(l)}z_{i_2}^{(l)}z_{i_3}^{(l)}z_{i_4}^{(l)}\rangle \rightarrow (\delta_{i_1 i_2}\delta_{i_3 i_4} + \delta_{i_1 i_3}\delta_{i_2 i_4} + \delta_{i_1 i_4}\delta_{i_2 i_3})\,\Big(G_2^{(l)}\Big)^2$$
This is exactly the four-point function you would expect from a totally Gaussian distribution. Look back at the pre-training chapter 2 to see this explicitly. Thus, in the infinite width limit, we can clearly see that our distribution becomes Gaussian (at least up to fourth order correlations).

To make things more realistic (infinite width isn't physical), let us consider the full expression and expand it in the large $n$ limit. Let us assume that all layer widths are equal so $n_l = n \,\forall\, l\in\{1,\ldots,L\}$. Let's now compute the deviation from Gaussianity in these four-point functions. 
$$G_4^{(l)} - \Big(G_2^{(l)}\Big)^2 = \Bigg[\,\Big(1 + \frac{2}{n}\Big)^{l-1}\,-1\Bigg]\,\Big(G_2^{(l)}\Big)^2$$
$$\therefore G_4^{(l)} - \Big(G_2^{(l)}\Big)^2\approx \frac{2\,(l-1)}{n}\,\Big(G_2^{(l)}\Big)^2 + \mathcal{O}\bigg(\frac{1}{n^2}\bigg)$$

**CODE TASK: VERIFY THIS SCALING EMPIRICALLY. SHOW THE DEVIATION OF THE FOUR-PT FUNCTION FROM GAUSSIANITY FOLLOWS A THIS SCALING WITH DEPTH  AND WIDTH.**

So, this correction factor scales inversely with width $n$ and proportionally with depth $l$, assuming we're in the critical limit where $G_2^{(l)}$ doesn't blow up or vanish with depth. This connects directly to the connected four-point correlator:
$$\langle z_{i_1}^{(l)}z_{i_2}^{(l)}z_{i_3}^{(l)}z_{i_4}^{(l)}\rangle|_{\textrm{connected}} = (\delta_{i_1 i_2}\delta_{i_3 i_4} + \delta_{i_1 i_3}\delta_{i_2 i_4} + \delta_{i_1 i_4}\delta_{i_2 i_3})\,\bigg(G_4^{(l)} - \Big(G_2^{(l)}\Big)^2\bigg)$$
Therefore, the non-Gaussianity of this DLN grows with depth. The quartic coupling thus grows with depth $l$, it **runs** as the layer changes. 

Another nice way to think about this connected four-point function is the interaction between different neurons in the same layer for $j\neq k$. 
$$\Big\langle\,\Big(z_j^{(l)}z_j^{(l)} - G_2^{(l)}\Big)\Big(z_k^{(l)}z_k^{(l)} - G_2^{(l)}\Big)\,\Big\rangle = G_4^{(l)} - \Big(G_2^{(l)}\Big)^2$$
where we are not taking an Einstein summation, we are just looking at neurons $j$ and $k$ in layer $l$. This other method of writing things down shows the deviation of $z_j z_j$ from its mean value $G_2$ and whether it is correlated to the deviation on a different neuron $k$. In the limit of infinite width, this deviation goes to 0. Thus, interactions among different neurons shut off in this limit. Note that in the large $n/L$ limit, the strength of these correlations grows with depth $l$. 

There are several other interpretations, but the big theme of this book is the following: 
	**All the finite width effects are proportional to the depth-to-width ratio $l/n$. The leading finite-width contributions at criticality grow linearly with depth $l$ and inversely with depth $n$.**

The deeper the network, the less the infinite-width Gaussian description works. This is because the deeper the network at finite width, the stronger the neuron-neuron correlations and the stronger the non-Gaussianity. Developing these non-Gaussian correlations is important for non-trivial feature learning. However, we don't want overly deep FCNs since there will be large fluctuations in observables such as $z_j^{(l)}z_j^{(l)}$ from instantiation to instantiation. We want non-trivial feature learning without the overly chaotic fluctuations in statistics. 

We will focus on the limit of large width ratio $n/L$, where perturbation theory works and we get the best of both worlds. In the opposite limit of large depth ratio $L/n$, perturbation theory totally breaks down. 

### 3.4 Chaos
Let's now compute higher order correlators to better understand the full distribution $p(z^{(l)}|\mathcal{D})$. We'll again assume a single data input $x_\alpha = x$. 

#### Math: recursion for six-point and higher-point correlators
We'll begin with the first layer. 
$$\langle z_{i_1}^{(1)}z_{i_2}^{(1)}\ldots z_{i_{2m}}^{(1)}\rangle = \langle W_{i_1 j_1}^{(1)} W_{i_2 j_2}^{(1)}\ldots W_{i_{2m} j_{2m}}^{(1)}\rangle\,x_{j_1} x_{j_2}\ldots x_{j_{2m}}$$
$$= \Bigg(\sum_{\textrm{all pairings}}  \delta_{i_{k_1} i_{k_2}}\ldots  \delta_{i_{k_{2m-1}} i_{k_{2m}}}\Bigg)\Big(C_W^{(1)}\Big)^m\,\Big(G_2^{(0)}\Big)^m$$
$$= \Bigg(\sum_{\textrm{all pairings}}  \delta_{i_{k_1} i_{k_2}}\ldots  \delta_{i_{k_{2m-1}} i_{k_{2m}}}\Bigg)\,\Big(G_2^{(1)}\Big)^m$$
where we've used our previous identities $G_{\alpha_1 \alpha_2}^{(0)} = \frac{1}{n_0}x_{j;\alpha_1}x_{j;\alpha_2}$ and $G_{\alpha_1\alpha_2}^{(l)} = \Big(\Pi_{i=1}^{l}C_W^{(i)}\Big)G_{\alpha_1\alpha_2}^{(0)}$. This is totally the result we expect for a pure Gaussian distribution, so we have the following action for the first layer:
$$S(z^{(1)}) = \frac{1}{2G_2^{(1)}}\,z_i^{(1)}z_i^{(1)}$$
where we're still using an Einstein summation. 

Now, we've already computed the four-point function generally for layer $l$. Let's now compute the six-point iteraction equation for the six-point function before moving to $2m$-point correlators more generally. 

$$\langle z_{i_1}^{(l)}z_{i_2}^{(l)}\ldots z_{i_{6}}^{(1)}\rangle = \langle W_{i_1 j_1}^{(l)} W_{i_2 j_2}^{(l)}\ldots W_{i_{6} j_{6}}^{(l)}\rangle\,\langle z_{j_1}^{(l-1)} z_{j_2}^{(l-1)}\ldots z_{j_{6}}^{(l-1)}\rangle$$
$$= {\Big(C_W^{(l)}\Big)^3}\, \Bigg(\sum_{\textrm{all pairings}}  \delta_{i_{k_1} i_{k_2}}\delta_{i_{k_3} i_{k_4}}\delta_{i_{k_{5}} i_{k_{6}}}\Bigg) \frac{1}{n_{l-1}^3}\langle z_{j_1}^{(l-1)} z_{j_1}^{(l-1)}z_{j_2}^{(l-1)}z_{j_{2}}^{(l-1)} z_{j_3}^{(l-1)}z_{j_{3}}^{(l-1)}\rangle$$
We can define the six-point correlator as follows (identically to how we've defined the previous correlators):
$$\langle z_{i_1}^{(l)}z_{i_2}^{(l)}z_{i_3}^{(l)}z_{i_4}^{(l)}z_{i_5}^{(l)} z_{i_{6}}^{(1)}\rangle = \Bigg(\sum_{\textrm{all pairings}}  \delta_{i_{k_1} i_{k_2}}\delta_{i_{k_3} i_{k_4}}\delta_{i_{k_{5}} i_{k_{6}}}\Bigg)\,G_6^{(l)}$$
Substituting this expression into the previous line where we have the expectation value of the six $z_{j_i}^{(l-1)}$. Since we contract pairs of indices in the previous line, and we have $\frac{6!}{(2!)^3(3!)} = 15$ unique pairings. You can write them all out and compute the sums to get the following six-point correlator:
$$G_6^{(l)} = \Big(C_W^{(l)}\Big)^3\,\Bigg(1 + \frac{6}{n_{l-1}} + \frac{8}{n_{l-1}^2}\Bigg)\,G_6^{(l-1)}$$
With this iterative relationship, we get the following simple solution:
$$G_6^{(l)} = \Bigg[\Pi_{l'=1}^{l-1}\,\Big(C_W^{(l')}\Big)^3\bigg(1 + \frac{6}{n_{l'}} + \frac{8}{n_{l'}^2}\bigg)\Bigg]\,G_6^{(0)}$$
We can use the initial condition that $G_6^{(0)} = (G_2^{(0)})^3$ to simplify things even further.
$$G_6^{(l)} = \Bigg[\Pi_{l'=1}^{l-1}\,\Big(C_W^{(l')}\Big)^3\bigg(1 + \frac{6}{n_{l'}} + \frac{8}{n_{l'}^2}\bigg)\Bigg]\,\Big(G_2^{(0)}\Big)^3$$
$$G_6^{(l)} = \Bigg[\Pi_{l'=1}^{l-1}\,\bigg(1 + \frac{6}{n_{l'}} + \frac{8}{n_{l'}^2}\bigg)\Bigg]\,\Big(G_2^{(l)}\Big)^3$$
where in the final line we've used the identity $\Big(G_2^{(l)}\Big)^3 = \Big(C_W^{(l')}\Big)^3\Big(G_2^{(0)}\Big)^3$. 

Great! We can thus clearly see the the product of factors encapsulate the deviation from Gaussianity. Basically, the factors $6/n_l$ and $8/n_l^2$ are terms that would not be present if we had a purely Gaussian distribution. 

Let us now generalise to arbitrary $2m$-point correlators at layer $l$. 
$$\langle z_{i_1}^{(l)}z_{i_2}^{(l)}\ldots z_{i_{2m}}^{(l)}\rangle = \Bigg(\sum_{\textrm{all pairings}}  \delta_{i_{k_1} i_{k_2}}\ldots  \delta_{i_{k_{2m-1}} i_{k_{2m}}}\Bigg)\,G_{2m}^{(l)}$$
We can then show that we get the following general recursion relation:
$$G_{2m}^{(l)} = \Big(C_W^{(l)}\Big)^m\,c_{2m}(n_{l-1})\,G_{2m}^{(l-1)}$$
where we have defined the following combinatorial factor:
$$c_{2m}(n) = \Big(1 + \frac{2}{n}\Big)\Big(1 + \frac{4}{n}\Big)\ldots \Big(1 + \frac{2(m-1)}{n}\Big) = \frac{(\frac{n}{2} -  1 + m)!}{(\frac{n}{2}-1)!}\,\bigg(\frac{2}{n}\bigg)^m$$
This comes from just adding up all the Wick contraction pairings and summing over all the contracted terms. **NOTE: I HAVEN'T DERIVED THIS MYSELF. I TRUST THE AUTHORS BUT TOO MUCH EFFORT TO REDERIVE RN...**

With this recursive relation, we have the following general expression:
$$G_{2m}^{(l)} = \bigg[\Pi_{l'=1}^{l-1}\,c_{2m}(n_{l'})\bigg]\,\Big(G_2^{(l)}\Big)^m$$

With this general expression, we're ready for the physics. As the authors say, "Enough with the math, time for the physics."

#### Physics: breakdown of perturbation theory and the emergence of chaos
To simplify things, let's assume constant width s.t. $n_l = n$ for all $l\geq 1$. We can say the following about our $2m$-point correlators.
- If we keep depth $L$ fixed but send width $n\rightarrow \infty$, all the combinatorial factors become unity. In this infinite width limit, we get the following result: $$G_{2m}^{(l)} = \Big(G_2^{(l)}\Big)^m$$ Thus, the output distribution $p(z^{(L)}|\mathcal{D})$ is just a Gaussian. So, the final output layer PDF is exactly the same as the first layer PDF: a basic Gaussian. 
- In the other regime where we set the depth $L\rightarrow \infty$ for fixed width $n$, then the combinatorial factors are all fixed and are greater than 1 at each layer. Thus, even for a critical network with $C_W^{(l)} = 1$, the $2m$-point correlator blows up progressively with depth $l$. This basically tells us that for an infinitely deep network, the correlators of the output are extremely non-Gaussian and very chaotic from instantiation to instantiation. 
- Thinking about these limits separately, they do not commute. $$\lim_{n\rightarrow\infty}\lim_{L\rightarrow \infty}G_{2m}^{(L)} \neq \lim_{L\rightarrow\infty}\lim_{n\rightarrow \infty}G_{2m}^{(L)}$$ because on the LHS, the correlation function blows up before we can go to infinite width but on the RHS, the distribution becomes boring and Gaussian so the infinite depth limit is also boring and Gaussian. Thus we have a chaotic and a boring limit. 
- We can construct an interpolating solution by sending both depth and width to infinity while keeping their ratio $r = L/n$ fixed. That is more meaningful. We can expand out the combinatorial factors in this limiting case (SEE THE TEXTBOOK FOR DETAILS) and we get the following: $$G_{2m}^{(L)}\rightarrow e^{m(m-1)r}\Big(G_2^{(L)}\Big)^m$$ With this, we see that looking at the limit of $L\rightarrow\infty$, the correlator blows up, as expected. For the limit $n\rightarrow \infty$, we get the basic Gaussian limit. 
- If we play around with this asymptotic exponential form a little, we get the following asymptotic 4-point connected correlator: $$G_4^{(L)} - \Big(G_2^{(L)}\Big)^2 = \Big(e^{2r}-1\Big)\Big(G_2^{(L)}\Big)^2= 2r\,\Big(G_2^{(L)}\Big)^2 + \mathcal{O}(r^2)$$ Clearly, this scales linearly with the ratio $r=L/n$. We can do the same for the six-point function. $$G_6^{(L)} - 3 G_2^{(L)}G_4^{(L)} + 2\Big(G_2^{(L)}\Big)^3 = \Big(e^{6r}-3e^{2r}+2\Big)\Big(G_2^{(L)}\Big)^2= 12r^2\,\Big(G_2^{(L)}\Big)^2 + \mathcal{O}(r^3)$$ Thus, this higher order connected correlator scales like $r^2$. Thus, as we go to higher order, the connected correlators obey nearly-Gaussian statistics with higher order interaction terms in the action become weaker and weaker in the wide limit. 