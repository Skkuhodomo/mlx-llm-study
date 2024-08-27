## Diffusion 모델 톺아보기 

### Overview

Diffusion 모델은 데이터에 노이즈를 점진적으로 추가하거나, 노이즈로부터 데이터를 점진적으로 복원하는 과정을 통해 데이터를 생성하는 모델이다. 이를 직관적으로 이해할 수 있도록 아래 그림을 참조하자. 아래 그림에서 \(x_0\)는 실제 데이터, \(x_T\)는 최종 노이즈, 그리고 그 사이의 \(x_t\)는 데이터에 노이즈가 더해진 상태의 잠재 변수(latent variable)를 나타낸다.
![diffusion](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM.png)
이미지 출처: Ho et al. 2020 with a few additional annotations
### Diffusion 모델의 구조

먼저, forward process \(q\)에서는 오른쪽에서 왼쪽으로 노이즈를 점진적으로 더해가며 데이터를 완전한 노이즈 상태로 변환한다. 이후 이 과정을 역으로 추정하는 reverse process \(p\)를 학습하여, 노이즈 \(x_T\)로부터 데이터 \(x_0\)를 복원하는 과정을 진행한다. 이 reverse process를 이용해 무작위 노이즈로부터 이미지, 텍스트, 그래프 등을 생성할 수 있는 모델을 구축할 수 있다. 궁극적으로, 이러한 과정을 통해 실제 데이터의 분포 \(p(x_0)\)를 추정하는 것이 목표다. 이제 이를 좀 더 자세히 살펴보겠다.

### Reverse Process

Reverse process \(p\)는 노이즈 \(x_T\)로부터 데이터를 복원하는 과정이다. 이는 무작위 노이즈로부터 데이터를 생성하는 모델로 사용되므로, 이를 모델링하는 것이 필수적이다. 하지만 이 과정은 매우 복잡하기 때문에, \(p_\theta\)를 이용하여 이를 근사한다. 이 근사는 가우시안 전이(Gaussian transition)를 사용한 마르코프 연쇄(Markov chain) 형태로 이루어진다. 수식으로 표현하면 다음과 같다:

\[p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod^T_{t=1} p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) \quad
p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))\]



이때 각 단계의 정규 분포의 평균 \(\mu_\theta\)와 표준편차 \(\Sigma_\theta\)는 학습해야 하는 파라미터다. 초기 노이즈 분포는 가장 단순한 형태의 표준정규분포로 정의된다.

### Forward Process

Forward process \(q\)는 데이터 \(x_0\)에 노이즈를 점진적으로 더해 최종 노이즈 상태인 \(x_T\)로 변환하는 과정이다. 이 과정의 분포를 파악하는 것이 중요한 이유는, reverse process의 학습을 위해 forward process의 정보를 사용하기 때문이다. Forward process는 reverse process와 유사하지만, 데이터에 가우시안 노이즈를 조금씩 더하는 마코프 연쇄 형태로 나타낸다. 이를 수식으로 표현하면 다음과 같다:
\[
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})\]

이러한 forward process의 장점은 데이터 \(x_0\)가 주어졌을 때 임의의 시간 단계 \(t\)에서 \(x_t\)를 자유롭게 샘플링할 수 있다는 점이다. 데이터 \(x_0\)가 주어졌을 때의 \(x_t\)의 분포는 다음과 같다:

\[
q(x_t | x_0) \sim \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)\mathbf{I})
\]

여기서 \(\bar{\alpha}_t\)는 다음과 같이 정의된다:

\[
\bar{\alpha}_t = \prod_{s=1}^t \alpha_s
\]

### Training

Diffusion 모델의 학습 손실은 정규 분포 간의 KL divergence 형태로 계산된다. 최종적으로, 우리는 아래와 같은 손실 함수로 모델을 학습시킨다:

\[
L = \mathbb{E}_q\left[\sum_{t=1}^T L_t \right]
\]

여기서 \(L_t\)는 \(p\)와 \(q\)의 reverse/forward process의 분포 차이를 나타내며, 각 단계에서 최대한 비슷하게 학습된다. 마지막으로 \(L_0\)는 잠재 변수 \(x_1\)으로부터 데이터 \(x_0\)를 추정하는 가능도(likelihood)로, 이를 최대화하는 방향으로 학습된다.

### 결론

이번 글에서는 Diffusion 모델의 개념과 학습 방법에 대해 소개하였다. 이 모델은 VAE, GAN에 이어 주목받는 또 다른 생성 모델로, 다양한 변형이 존재한다. 향후 DDPM, D3PM, NCSN 등 다양한 변형 모델에 대해서도 다룰 예정이다. 계속해서 새로운 내용을 공유할 계획이니 기대해 달라. 

학업으로 바쁜 가운데 블로그 글을 작성하는 것이 쉽지 않지만, 앞으로도 지속적으로 관련 내용을 정리하여 공유하겠다.

### 참조
[1] https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
