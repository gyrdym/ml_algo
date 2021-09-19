/// A type of learning rate strategies
enum LearningRateType {
  /// The type is deprecated, use [LearningRateType.timeBased] instead
  @deprecated
  decreasingAdaptive,

  /// Learning rate value will be constant throughout the whole fitting process
  constant,

  /// Learning rate value will be calculated according to the formula:
  ///
  /// ![\[\bg_white \eta_{n + 1}= \frac{\eta _{n}}{1+dn}\]](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D%20%5Cbg_white%20d%5Ceta_%7Bn%20&plus;%201%7D=%20%5Cfrac%7B%5Ceta%20_%7Bn%7D%7D%7B1&plus;dn%7D%5C)
  ///
  /// where:
  ///
  /// ![\[\bg_white \eta_{n + 1}\]](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D%20%5Cbg_white%20%5C%5B%5Ceta_%7Bn%20&plus;%201%7D%5C) is the learning rate value for a new iteration
  ///
  /// ![\[\bg_white \eta_{n}\]](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D%20%5Cbg_white%20%5Ceta_%7Bn%7D) is the learning rate value from a previous iteration
  ///
  /// ![\[\bg_white d\]](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D%20%5Cbg_white%20d) is the decay parameter
  ///
  /// ![\[\bg_white n\]](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D%20%5Cbg_white%20n) is the iteration step
  timeBased,

  /// Learning rate value will be calculated according to the formula:
  ///
  /// ![\[\bg_white \eta_{n}= \eta _{0}d^{floor(\frac{1 + n}{r})} \eta_{n}\]](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D%20%5Cbg_white%20%5Ceta_%7Bn%7D=%20%5Ceta%20_%7B0%7Dd%5E%7Bfloor(%5Cfrac%7B1%20&plus;%20n%7D%7Br%7D)%7D%20)
  ///
  /// where:
  ///
  /// ![\[\bg_white \eta_{n}\]](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D%20%5Cbg_white%20%5Ceta_%7Bn%7D) is the new learning rate value
  ///
  /// ![\[\bg_white \eta_{0}\]](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D%20%5Cbg_white%20%5Ceta_%7B0%7D) is the initial learning rate value
  ///
  /// ![\[\bg_white d\]](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D%20%5Cbg_white%20d) is the decay parameter (in the context of step-based strategy, it describes, how much the learning rate should change at each drop, e.g. 0.5 means a halving)
  ///
  /// ![\[\bg_white n\]](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D%20%5Cbg_white%20n) is the iteration step
  ///
  /// ![\[\bg_white r\]](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D%20%5Cbg_white%20r) is the drop rate (how often the learning rate value should be dropped, r=5 means the rate will be dropped every 5 iterations)
  stepBased,

  /// Learning rate value will be calculated according to the formula:
  ///
  /// ![\[\bg_white \eta_{n}=\eta_{0}e^{-dn}\]](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D%20%5Cbg_white%20%5Ceta_%7Bn%7D=%5Ceta_%7B0%7De%5E%7B-dn%7D)
  ///
  /// where:
  ///
  /// ![\[\bg_white \eta_{n}\]](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D%20%5Cbg_white%20%5Ceta_%7Bn%7D) is the new learning rate value
  ///
  /// ![\[\bg_white \eta_{0}\]](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D%20%5Cbg_white%20%5Ceta_%7B0%7D) is the initial learning rate value
  ///
  /// ![\[\bg_white d\]](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D%20%5Cbg_white%20d) is the decay parameter
  ///
  /// ![\[\bg_white n\]](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D%20%5Cbg_white%20n) is the iteration step
  exponential,
}

const defaultLearningRateType = LearningRateType.constant;
