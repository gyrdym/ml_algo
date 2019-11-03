void validateTreeSolverMaxDepth(int maxDepth) {
  if (maxDepth < 0) {
    throw Exception('Maximal tree depth value should be greater than zero, but '
        '$maxDepth given');
  }
}
