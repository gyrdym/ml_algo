/// A type of learning rate behaviour
enum LearningRateType {
  /// The type is deprecated, use [LearningRateType.timeBased] instead
  @deprecated
  decreasingAdaptive,

  /// Learning rate will be constant throughout the whole fitting process
  constant,

  timeBased,

  stepBased,

  exponential,
}

const defaultLearningRateType = LearningRateType.constant;
