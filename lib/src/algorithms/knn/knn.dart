import 'package:ml_algo/src/algorithms/knn/neigbour.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:quiver/iterables.dart';

/// Finds [k] nearest neighbours for either observation in [observations]
/// basing on [trainObservations] and [trainOutcomes]
Iterable<Iterable<Neighbour<Vector>>> findKNeighbours(int k,
    Matrix trainObservations,
    Matrix trainOutcomes,
    Matrix observations,
) sync* {
  final allNeighbours = zip([trainObservations.rows, trainOutcomes.rows]);
  final firstKNeighbours = allNeighbours.take(k);
  final restNeighbours = allNeighbours.skip(k);
  for (final observation in observations.rows) {
    final sortedKNeighbors = firstKNeighbours
        .map((pair) => Neighbour(pair.first.distanceTo(observation), pair.last))
        .toList(growable: false)
        ..sort((pair1, pair2) => (pair1.distance - pair2.distance) ~/
                (pair1.distance - pair2.distance).abs());
    restNeighbours.forEach((pair) {
      final newKNeighbour = Neighbour(observation.distanceTo(pair.first),
          pair.last);
      final newNeighbourIdx = _findNewNeighbourIdx(newKNeighbour.distance,
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

int _findNewNeighbourIdx(double newNeighbourDist,
    List<Neighbour> sortedNeighbors) {
  var i = -1;
  for (final neighbour in sortedNeighbors) {
    if (newNeighbourDist < neighbour.distance) return ++i;
  }
  return i;
}