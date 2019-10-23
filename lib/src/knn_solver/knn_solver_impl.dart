import 'package:ml_algo/src/knn_solver/knn_solver.dart';
import 'package:ml_algo/src/knn_solver/neigbour.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:quiver/iterables.dart';

class KnnSolverImpl implements KnnSolver {
  KnnSolverImpl(
      this._trainFeatures,
      this._trainOutcomes,
      this._k,
      this._distanceType,
      this._standardize,
  ) {
    if (!_trainFeatures.hasData) {
      throw Exception('Empty features matrix provided');
    }
    if (!_trainOutcomes.hasData) {
      throw Exception('Empty outcomes matrix provided');
    }
    if (_trainOutcomes.columnsNum > 1) {
      throw Exception('Invalid outcome matrix: it is expected to be a column '
          'vector, but a matrix of ${_trainOutcomes.columnsNum} colums is '
          'given');
    }
    if (_trainFeatures.rowsNum != _trainOutcomes.rowsNum) {
      throw Exception('Number of feature records and number of associated '
          'outcomes must be equal');
    }
    if (_k <= 0 || _k > _trainFeatures.rowsNum) {
      throw RangeError.value(_k, 'Parameter k should be within the range '
          '1..${_trainFeatures.rowsNum} (both inclusive)');
    }
  }

  final Matrix _trainFeatures;
  final Matrix _trainOutcomes;
  final int _k;
  final Distance _distanceType;
  final bool _standardize;

  @override
  Iterable<Iterable<Neighbour<Vector>>> findKNeighbours(Matrix features) {
    if (!features.hasData) {
      throw Exception('No features provided');
    }

    if (features.columnsNum != _trainFeatures.columnsNum) {
      throw Exception('Invalid feature matrix: expected columns number: '
          '${_trainFeatures.columnsNum}, given: '
          '${features.columnsNum}');
    }

    final allNeighbours = zip([_trainFeatures.rows, _trainOutcomes.rows]);
    final firstKNeighbours = allNeighbours.take(_k);
    final restNeighbours = allNeighbours.skip(_k);

    return features.rows.map((observation) {
      final sortedKNeighbors = firstKNeighbours
          .map((pair) => Neighbour(pair.first
          .distanceTo(observation, distance: _distanceType), pair.last))
          .toList(growable: false)
        ..sort((pair1, pair2) => (pair1.distance - pair2.distance) ~/ 1);

      restNeighbours.forEach((pair) {
        final newKNeighbour = Neighbour(observation.distanceTo(pair.first),
            pair.last);
        final newNeighbourIdx = _findNewNeighbourIdx(newKNeighbour.distance,
            sortedKNeighbors);
        if (newNeighbourIdx != -1) {
          sortedKNeighbors.setRange(newNeighbourIdx + 1, _k, sortedKNeighbors,
              newNeighbourIdx);
          sortedKNeighbors[newNeighbourIdx] = newKNeighbour;
        }
      });

      if (_standardize && sortedKNeighbors.isNotEmpty) {
        final distanceOfLast = sortedKNeighbors.last.distance;
        return sortedKNeighbors.map((neighbour) =>
            Neighbour(neighbour.distance / distanceOfLast, neighbour.label));
      }

      return sortedKNeighbors;
    });
  }

  int _findNewNeighbourIdx(double newNeighbourDist,
      List<Neighbour> sortedNeighbors) {
    var i = -1;
    for (final neighbour in sortedNeighbors) {
      if (newNeighbourDist < neighbour.distance) return ++i;
    }
    return i;
  }
}
