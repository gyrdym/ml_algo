void validateProbabilityValue(num probability) {
  if (probability < 0 || probability > 1) {
    throw RangeError.range(probability, 0, 1, 'wrong probability value');
  }
}
