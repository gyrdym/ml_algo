import 'package:ml_algo/src/classifier/decision_tree_classifier/helpers/svg/get_tree_width.dart';

Map<int, num> getTreeNodeDistanceByLevel<T>(
    List<List<T>> levels, num nodeWidth, num minDist) {
  if (levels.length == 1) {
    return {0: minDist};
  }

  final lastLevelIdx = levels.length - 1;
  final totalWidth = getTreeWidth(levels, nodeWidth, minDist);
  final distByLevel = <int, num>{lastLevelIdx: minDist};

  for (var i = 0; i < levels.length - 1; i++) {
    distByLevel[i] =
        (totalWidth - (levels[i].length * nodeWidth)) / levels[i].length;
  }

  return distByLevel;
}
