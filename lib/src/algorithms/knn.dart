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
  final firstKNeighbours = allNeighbours.take(k).toList(growable: false);
  final restNeighbours = allNeighbours.skip(k);
  for (final observation in observations.rows) {
    final kNeighbours = firstKNeighbours
        .map((pair) => Tuple2(pair.first.distanceTo(observation), pair.last))
        .toList(growable: false);
    restNeighbours.forEach((pair) {
      final newKNeighbour = Tuple2(observation.distanceTo(pair.first),
          pair.last);
      for (int i = k - 1; i >= 0; i--) {
        if (newKNeighbour.item1 < kNeighbours[i].item1) {
          kNeighbours.setRange(i + 1, k, kNeighbours, i);
          kNeighbours[i] = newKNeighbour;
          break;
        }
      }
    });
    yield kNeighbours;
  }
}