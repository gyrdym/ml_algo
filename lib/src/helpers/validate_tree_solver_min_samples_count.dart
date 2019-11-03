void validateTreeSolversMinSamplesCount(int minSamplesCount) {
  if (minSamplesCount < 0) {
    throw Exception('Minimal samples count should be greater than zero, but '
        '$minSamplesCount given');
  }
}
