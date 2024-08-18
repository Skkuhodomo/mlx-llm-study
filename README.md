##  MLX ? 
MLX란, 넘파이와 비슷한, 프레임워크입니다. 이는 Apple-silicon에서의 효율적이고 유연한 머신러닝을 위하여 설게되었습니다. 
MLX의 파이썬 API는 일부 예외를 제외하고는 거의 넘파이와 유사합니다. MLX는 또한, 파이썬과 유사한 C++ API 역시 제공합니다.

아래는 넘파이와 MLX의 큰 차이입니다.
> - 함수 변환 조합 가능성(Composable function transformations): MLX는 자동 미분, 자동 벡터화 및 계산 그래프 최적화를 위한 함수 변환을 조합할 수 있습니다.
> 
> - 지연 계산(Lazy Computation) : MLX에서의 계산은 지연 계산으로 수행됩니다. 배열은 실제로 필요할 때만 생성됩니다.
> 
> - Multi-device: 연산은 지원되는 모든 장치(CPU, GPU)에서 가능합니다.

MLX의 프레임워크는 PyTorch, Jax, 그리고 ArrayFire에 영감 받았습니다. MLX의 가장 큰 차이점은, "MLX is the unified memory model.", 통합 메모리 모델이라는 점입니다. 

## This Repository...
이 레포지토리는 MLX의 기본적인 연산을 학습하고, 궁극적으로 Swift까지 이어져, 온-디바이스 LLM을 구축하는 것을 목표로 합니다. 

## Reference 
개인이 학습하면서 만든 공간으로, 제가 볼 자료는 MLX의 공식 문서입니다. https://ml-explore.github.io/mlx/build/html/index.html
