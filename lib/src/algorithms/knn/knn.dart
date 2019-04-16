import 'package:ml_algo/src/algorithms/knn/neigbour.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:quiver/iterables.dart';

typedef FindKnnFn = Iterable<Iterable<Neighbour<Vector>>> Function(int k,
    Matrix trainObservations, Matrix labels, Matrix observations,
    {Distance distance});

/// Finds [k] nearest neighbours for either observation in [observations]
/// basing on [trainObservations] and [labels]
Iterable<Iterable<Neighbour<Vector>>> findKNeighbours(int k,
    Matrix trainObservations,
    Matrix labels,
    Matrix observations,
    {
      Distance distance = Distance.euclidean,
    }
) sync* {
  final allNeighbours = zip([trainObservations.rows, labels.rows]);
  final firstKNeighbours = allNeighbours.take(k);
  final restNeighbours = allNeighbours.skip(k);
  for (final observation in observations.rows) {
    final sortedKNeighbors = firstKNeighbours
        .map((pair) => Neighbour(pair.first
        .distanceTo(observation, distance: distance), pair.last))
        .toList(growable: false)
        ..sort((pair1, pair2) => (pair1.distance - pair2.distance) ~/ 1);
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
