void validateProbabilityValue(num probability, {double precision = 1e-5}) {
  if (probability + precision < 0 || probability - precision > 1) {
    throw RangeError.range(probability, 0, 1, 'wrong probability value');
  }
}
