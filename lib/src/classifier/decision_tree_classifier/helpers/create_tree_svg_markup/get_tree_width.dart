num getTreeWidth<T>(List<List<T>> levels, num nodeWidth, num minDist) {
  final lastLevel = levels[levels.length - 1];

  return lastLevel.length * (nodeWidth + minDist);
}
