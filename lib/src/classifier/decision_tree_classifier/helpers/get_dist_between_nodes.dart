class _Return {
  _Return(this.totalWidth, this.distByLevel);

  final num totalWidth;
  final Map<int, num> distByLevel;
}

_Return getDistBetweenNodes<T>(List<List<T>> levels, num nodeWidth, num minDist) {
  if (levels.length == 1) {
    return _Return(nodeWidth + minDist, {0: minDist});
  }

  final lastLevelIdx = levels.length - 1;
  final lastLevel = levels[lastLevelIdx];
  final totalWidth = lastLevel.length * (nodeWidth + minDist);
  final distByLevel = <int, num>{lastLevelIdx: minDist};

  for (var i = 0; i < levels.length - 1; i++) {
    distByLevel[i] = (totalWidth - (levels[i].length * nodeWidth)) / levels[i].length;
  }

  return _Return(totalWidth, distByLevel);
}
