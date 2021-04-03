/// A type of learning rate behaviour
enum LearningRateType {
  /// Learning rate will decrease every iteration according to the rule:
  ///
  /// ````dart
  /// learningRateValue / iterationNumber
  /// ````
  decreasingAdaptive,

  /// Learning rate will be constant throughout the whole fitting process
  constant,
}

const defaultLearningRateType = LearningRateType.decreasingAdaptive;
