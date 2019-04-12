import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:quiver/iterables.dart';
import 'package:tuple/tuple.dart';

/// Finds [k] nearest neighbours for either observation in [observations]
/// basing on [trainObservations] and [trainOutcomes]
Iterable<Iterable<Tuple2<double, Vector>>> findKNeighbours(int k,
    Matrix trainObservations,
    Matrix trainOutcomes,
    Matrix observations,
) sync* {
  final allNeighbours = zip([trainObservations.rows, trainOutcomes.rows]);
  final firstKNeighbours = allNeighbours.take(k);
  final restNeighbours = allNeighbours.skip(k);
  for (final observation in observations.rows) {
    final sortedKNeighbors = firstKNeighbours
        .map((pair) => Tuple2(pair.first.distanceTo(observation), pair.last))
        .toList(growable: false)
        ..sort((pair1, pair2) => (pair1.item1 - pair2.item1) ~/
                (pair1.item1 - pair2.item1).abs());
    restNeighbours.forEach((pair) {
      final newKNeighbour = Tuple2(observation.distanceTo(pair.first),
          pair.last);
      final newNeighbourIdx = _findNewNeighbourIdx(newKNeighbour.item1, k,
          sortedKNeighbors);
      if (newNeighbourIdx != -1) {
        sortedKNeighbors.setRange(newNeighbourIdx + 1, k, sortedKNeighbors,
            newNeighbourIdx);
        sortedKNeighbors[newNeighbourIdx] = newKNeighbour;
      }
    });
    yield sortedKNeighbors;
  }
}

int _findNewNeighbourIdx(double newNeighbourDist, int k,
    List<Tuple2<double, Vector>> sortedKNeighbors) {
  for (int i = 0; i < k; i++) {
    if (newNeighbourDist < sortedKNeighbors[i].item1) return i;
  }
  return -1;
}