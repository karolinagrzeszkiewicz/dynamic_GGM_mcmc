# Dynamic Graphical Models MCMC

This Particle MCMC algorithm is aimed at recovering the changing correlation structure from timeseires data via Bayesian inference on undirected graphs. 
The correlations between variables can be represented as edges between nodes – in case the variables are correlated there is an edge (corresponding to weight of an edge
equal to 1), and in case they are not, there is no edge (corresponding to a weight of 0). Then for timeseries data there can be changepoints i.e. timesteps when the graph changes.
This model and algorithm approximates the joint posterior distribution of changepoints and the graphs, using a particle filter algorithm to approximate the 
likelihood of data given a list of changepoints.

The model has the following parameters:
```
struct hyper_pars
    p::Int ## no of nodes
  Tau::Int ## no of time instances
    w::Float64 ## w in (0,1], gives E[no of edges]=p*w for G0
    z::Float64 ## z in (0,1], gives E[no of edges flips]=p*z for G|Gp
delta::Float64 ## delta>2 is parameter of G-Wishart(delta, D)
  tau::Float64 ## tau>0 is s.t. D=tau*I in G-Wishart(delta,D)
theta::Float64 # in [0,1], parameter of Geometric prior
    g::Float64 #hyperparameter for geometric proposal
end
#############################
struct MCMC_pars
      M::Int ## no of MCMC steps
     pB::Float64 ## p(Birth)
end
#############################
struct PF_pars
    N::Int ## no of particles
 ESS0::Float64 ## ESS0 in (0,1) is threshold for ESS triggering resampling
  TOL::Float64 ## tolerance level for bisection method convergence
   nE::Int ## no of edges flipped at mutation step
   nM::Int ## no of single MCMC steps synthesized to make up a mutation step
   num_it::Int ## no of MC interations for normalising constant
   Laplace::Bool ## true = use the Laplace method for computing norm const
   is_parallel::Bool
   master :: Int
end
```

## How to run the code?
The simulations folder contains some simulation scripts in .jl or .ipynb format. To run the .jl simulations e.g. run
```
julia simulations/0changepoints_small.jl 
```
from the dynamic_GGM_mcmc directory.


## The model

### Prior on graphs:
Prior for graph with no parent P(G):
Let A1,A_2,...,Ap(p−1)$ be all of the possible edges of G, a graph with p nodes. Then for every i > 1
1≤i≤ p(p−1) Ai ∼Bernoulli(2w/p-1)

Prior for graph with parent P(G|Gp):
Let A1,A2,...,Ap(p−1) be the edges of G and Ap1,Ap2,...,App(p−1) the edges of Gp, the parent
of G.Then for every i1≤i≤p(p−1) |Ai−Api|∼Bernoulli(2w/p-1)

### Prior on changepoints:
First, we specify a collection of random changepoints c1, c2, . . . , cκ, for κ ≥ 0, across the discrete time instances {1,...,T}. The prior distribution p(c1:κ) is chosen as
κ ∼ Geometric(mean = (1 − p0)/p0), c1:κ|κ ∼ Uniform(Tκ,l) so that Tκ,l is the space of ordered κ-tuples c1 < c2 < ··· < cκ in {1,2,...,T}, 
under the minimum-duration restriction that for every cj+1 − cj ≥ l, j = 0, 1, . . . , κ.

### The likelihood on graphs:
Can be obtained via a G-Wishart distribution of precision matrices given graphs, Then the likelihood of data between two changepoints c_j and c_(j+1)
p(Yc_j:c_(j+1)−1|Gc_j) can be obtained by integrating out precision matrices Ω1:T in [Yc_j:c_(j+1)−1 |Ω1:T,G1:T,c1:κ ] ∼ N(0,Ωcj). As this is computationally intractable, 
we use Laplace approximation to approximate the likehood dsitribution.

### The likelihood on changepoints:
Given changepoints C we have a Hidden Markov Model, with signal given by G0:T , and observations Y0:T . 
Given the simple structure in the transitions of the signal, we can apply a filtering method to obtain the density p(y0:T | C).

The partilce filter algorithm:

```
Step 1: Initalise
For all i ∈ {1, 2, ...N } set Gi and Gip equal to pxp graphs of zeros, set the unnormalised weights wi equal to 1. Initalialise P(G|Gp) and P(Y|G) with the value of 1. Initialise ˆ(r) (estimate of
P (Y |S)) with the value of 0. 

Step 2: For l in 1: k+1:
1.Fori∈{1,2,...N}sampleGi fromthegraphprior(i.e. forj=1Gi ∼P(G),forj≥2 Gi ∼ P(G|Gp)), slice the data to get Yc = {YSl−1,...,YSl−1} where S0 = 1 and Sk+1 = T
2. Let tl be the temperature at l (obtained via tempering), initialise φc, φn = 0
3. For j in 1:t:
(a) Set φc = φn
(b) Fori∈{1,2,...N}calculateP(Y|Gi)
(c) Set φn = tl,j
(d) Fori∈{1,2,...N}setwi :=wiP(Y|Gi)φn−φc
(e) If φn < 1
i.Estimate:rˆ:=rˆ×N1 􏰆Ni=1wi
ii. Resample: Let Wi for 1 ≤ i ≤ N be the normalised weights. Sample N particles from
the categorical distribution where particle i is sampled with probability Wi, calculate P(Yc|Gi) for each of the sampled particles. Set G equal to the resampled particles and Gp equal to their parents.
iii. Mutate: mutate each sampled Gi based on likelihood P (Yc|Gi) and P (Gi|Gip), for all i,1≤i≤N setwi :=1
4. Set rˆ := ˆ(r) × N1 􏰆Ni=1 wi
5. For i ∈ {1,2,...N} set Gip := Gi

Step 3: Sample a particle from the categorical distribution where particle i is sampled with probability Wi and obtain its history. Return rˆ and the sequence of graphs for the sampled particle.
```

Note that this algorithm also returns a posterior distribution on graphs given the proposed inout list of changepoints, as the graphs are filtered based on their prior and likelihood.

### The posterior:
is obtained by applying Partilce MCMC. At every iteration a list of changepoints is sampled from a proposal (as a birth/death of changepoint from the previous 
accepted list of changepoints, or as a shoft of a changepoint). Its likelihood is computed using paricle filter, and its prior is computed, which (assuming we know the likelihood and prior for the previous 
list of changepoints) gives us the acceptance probability of this proposal. Then the proposal is accepted with that probability, and if it is accepted the posterior distribution of graphs
given these changepoints (computed by the partcile filter) is accepted too.

## Example:

Setting:
```
hyper = hyper_pars(4, 100, 0.5, 0.5, 3.0, 1.0/3.0, 0.5)
#(p = 4, Tau = 100, w = 1, z = 1, delta = 3, tau = 1/3, theta =0.5 )
parPF = PF_pars(200, 0.7, 0.001, 3, 20, 1000, true)
#(N, ESS0, TOL, nE, nM, num_it, false)
#changed N from 10 to 1, again to 1000
parMCMC = MCMC_pars(200, 0.5)
# (M, pB)
```

we run the algorithm on data consisting of 100 datapoints and 4 variables generated from multivariate normal distribution with mean vectors of zeros 
and (1) precision matrix with diagonal entries equal to 1, off-diagonal entries 0.6, and all other entries equal to 0 until timestep 48 (changepoint) 
(2) same as the previous precision matrix but with the second off-diagonal row entries equal to 0.7 from timestep 48 onwards.

Then initialising the algorithm with the proposal of a single changepoint at 9 we get the following output, which shows how the algorithm explores the spcae of 
changepoints but eventually converges to the single correct changepoint at 48 as this is the proposal favoured by the prior and the likelihood:
```
Sc = [9]
logP(Y|Sc) = -829.7373704299875
Graph = [0.0 0.9586264735551696 0.02415217402984606 0.2275961018483285; 0.0 0.0 0.8893414133452203 0.7436618386638694; 0.0 0.0 0.0 0.3269712311904606; 0.0 0.0 0.0 0.0]
proposal Spr = [29]
proposal type = Move
logP(Y|Spr) =-816.4328250360386
log acceptance probability = 13.356505132879668
accepted
proposal Spr = [71]
proposal type = Move
logP(Y|Spr) =-804.0664204784924
log acceptance probability = 12.366404557546161
accepted
proposal Spr = [40]
proposal type = Move
logP(Y|Spr) =-796.826658099979
log acceptance probability = 7.239762378513433
accepted
proposal Spr = [10]
proposal type = Move
logP(Y|Spr) =-830.0714891509125
log acceptance probability = -33.28405176408678
rejected
proposal Spr = [86]
proposal type = Move
logP(Y|Spr) =-816.6695456427575
log acceptance probability = -19.8428875427785
rejected
proposal Spr = [65]
proposal type = Move
logP(Y|Spr) =-792.9885644220157
log acceptance probability = 3.838093677963343
accepted
proposal Spr = [19]
proposal type = Move
logP(Y|Spr) =-824.9540257740503
log acceptance probability = -31.9654613520346
rejected
proposal Spr = [94]
proposal type = Move
logP(Y|Spr) =-822.2269789721781
log acceptance probability = -29.315375591298608
rejected
proposal Spr = [41]
proposal type = Move
logP(Y|Spr) =-796.1142722054806
log acceptance probability = -3.125707783464918
rejected
proposal Spr = [79]
proposal type = Move
logP(Y|Spr) =-812.8141019677296
log acceptance probability = -19.825537545713928
rejected
proposal Spr = [66]
proposal type = Move
logP(Y|Spr) =-793.6057224604742
log acceptance probability = -0.6171580384585695
rejected
proposal Spr = [48]
proposal type = Move
logP(Y|Spr) =-772.9519339120636
log acceptance probability = 20.036630509952033
accepted
proposal Spr = [49]
proposal type = Move
logP(Y|Spr) =-777.2903132195012
log acceptance probability = -4.338379307437549
rejected
proposal Spr = [25]
proposal type = Move
logP(Y|Spr) =-816.868414097549
log acceptance probability = -43.9164801854854
rejected
proposal Spr = [56]
proposal type = Move
logP(Y|Spr) =-792.7479048078245
log acceptance probability = -19.795970895760888
rejected
proposal Spr = [72]
proposal type = Move
logP(Y|Spr) =-805.4866234465799
log acceptance probability = -32.53468953451625
rejected
proposal Spr = [46]
proposal type = Move
logP(Y|Spr) =-776.2749166963861
log acceptance probability = -3.3229827843224484
rejected
proposal Spr = [30]
proposal type = Move
logP(Y|Spr) =-816.7969674008907
log acceptance probability = -43.845033488827085
rejected
proposal Spr = [34]
proposal type = Move
logP(Y|Spr) =-811.6900959349423
log acceptance probability = -38.73816202287867
rejected
proposal Spr = [20]
proposal type = Move
logP(Y|Spr) =-825.7203140034828
log acceptance probability = -52.7683800914192
rejected
proposal Spr = [13]
proposal type = Move
logP(Y|Spr) =-828.7563741237354
log acceptance probability = -55.80444021167182
rejected
proposal Spr = [26]
proposal type = Move
logP(Y|Spr) =-817.2564948911624
log acceptance probability = -44.304560979098824
rejected
proposal Spr = [89]
proposal type = Move
logP(Y|Spr) =-820.7186441731144
log acceptance probability = -47.7799554878008
rejected
proposal Spr = [48]
proposal type = Move
logP(Y|Spr) =-773.0105313302597
log acceptance probability = -0.05859741819608644
accepted
proposal Spr = [81]
proposal type = Move
logP(Y|Spr) =-812.5191150878487
log acceptance probability = -39.50858375758901
rejected
proposal Spr = [59]
proposal type = Move
logP(Y|Spr) =-794.1605241168141
log acceptance probability = -21.149992786554435
rejected
proposal Spr = [92]
proposal type = Move
logP(Y|Spr) =-822.0065595953826
log acceptance probability = -49.04798800405356
rejected
proposal Spr = [90]
proposal type = Move
logP(Y|Spr) =-820.3222693525302
log acceptance probability = -47.33805533058782
rejected
proposal Spr = [10]
proposal type = Move
logP(Y|Spr) =-829.9304051121827
log acceptance probability = -56.959094495076286
rejected
proposal Spr = [68]
proposal type = Move
logP(Y|Spr) =-796.0005839474433
log acceptance probability = -22.99005261718355
rejected
proposal Spr = [60]
proposal type = Move
logP(Y|Spr) =-796.5148652796335
log acceptance probability = -23.504333949373745
rejected
proposal Spr = [7]
proposal type = Move
logP(Y|Spr) =-825.413027461189
log acceptance probability = -52.47945717206544
rejected
proposal Spr = [91]
proposal type = Move
logP(Y|Spr) =-821.1222688943277
log acceptance probability = -48.15095827722128
rejected
proposal Spr = [90]
proposal type = Move
logP(Y|Spr) =-820.1918231581551
log acceptance probability = -47.20760913621277
rejected
proposal Spr = [15]
proposal type = Move
logP(Y|Spr) =-823.6818130199707
log acceptance probability = -50.67128168971101
rejected
proposal Spr = [71]
proposal type = Move
logP(Y|Spr) =-803.9779786648301
log acceptance probability = -30.967447334570355
rejected
proposal Spr = [62]
proposal type = Move
logP(Y|Spr) =-792.0687304480888
log acceptance probability = -19.058199117829076
rejected
```
etc. (the remaining output was trimmed off)
