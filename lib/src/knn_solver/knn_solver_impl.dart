import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/common/constants/common_json_keys.dart';
import 'package:ml_algo/src/common/json_converter/distance_type_json_converter.dart';
import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';
import 'package:ml_algo/src/knn_solver/knn_solver.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_constants.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_json_keys.dart';
import 'package:ml_algo/src/knn_solver/neigbour.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:quiver/iterables.dart';

part 'knn_solver_impl.g.dart';

@JsonSerializable()
@DistanceTypeJsonConverter()
class KnnSolverImpl with SerializableMixin implements KnnSolver {
  KnnSolverImpl(
    this.trainFeatures,
    this.trainOutcomes,
    this.k,
    this.distanceType,
    this.standardize, {
    this.schemaVersion = knnSolverJsonSchemaVersion,
  }) {
    if (!trainFeatures.hasData) {
      throw Exception('Empty features matrix provided');
    }

    if (!trainOutcomes.hasData) {
      throw Exception('Empty outcomes matrix provided');
    }

    if (trainOutcomes.columnsNum > 1) {
      throw Exception('Invalid outcome matrix: it is expected to be a column '
          'vector, but a matrix of ${trainOutcomes.columnsNum} colums is '
          'given');
    }

    if (trainFeatures.rowsNum != trainOutcomes.rowsNum) {
      throw Exception('Number of feature records and number of associated '
          'outcomes must be equal');
    }

    if (k <= 0 || k > trainFeatures.rowsNum) {
      throw RangeError.value(
          k,
          'Parameter k should be within the range '
          '1..${trainFeatures.rowsNum} (both inclusive)');
    }
  }

  factory KnnSolverImpl.fromJson(Map<String, dynamic> json) =>
      _$KnnSolverImplFromJson(json);

  @override
  Map<String, dynamic> toJson() => _$KnnSolverImplToJson(this);

  @JsonKey(name: knnSolverTrainFeaturesJsonKey)
  final Matrix trainFeatures;

  @JsonKey(name: knnSolverTrainOutcomesJsonKey)
  final Matrix trainOutcomes;

  @override
  @JsonKey(name: knnSolverKJsonKey)
  final int k;

  @override
  @JsonKey(name: knnSolverDistanceTypeJsonKey)
  final Distance distanceType;

  @JsonKey(name: knnSolverStandardizeJsonKey)
  final bool standardize;

  @override
  @JsonKey(name: jsonSchemaVersionJsonKey)
  final schemaVersion;

  @override
  Iterable<Iterable<Neighbour<Vector>>> findKNeighbours(Matrix features) {
    if (!features.hasData) {
      throw Exception('No features provided');
    }

    if (features.columnsNum != trainFeatures.columnsNum) {
      throw Exception('Invalid feature matrix: expected columns number: '
          '${trainFeatures.columnsNum}, given: '
          '${features.columnsNum}');
    }

    final allNeighbours = zip([trainFeatures.rows, trainOutcomes.rows]);
    final firstKNeighbours = allNeighbours.take(k);
    final restNeighbours = allNeighbours.skip(k);

    return features.rows.map((observation) {
      final sortedKNeighbors = firstKNeighbours
          .map((pair) => Neighbour(
              pair.first.distanceTo(observation, distance: distanceType),
              pair.last))
          .toList(growable: false)
            ..sort((pair1, pair2) => (pair1.distance - pair2.distance) ~/ 1);

      restNeighbours.forEach((pair) {
        final newKNeighbour =
            Neighbour(observation.distanceTo(pair.first), pair.last);
        final newNeighbourIdx =
            _findNewNeighbourIdx(newKNeighbour.distance, sortedKNeighbors);
        if (newNeighbourIdx != -1) {
          sortedKNeighbors.setRange(
              newNeighbourIdx + 1, k, sortedKNeighbors, newNeighbourIdx);
          sortedKNeighbors[newNeighbourIdx] = newKNeighbour;
        }
      });

      if (standardize && sortedKNeighbors.isNotEmpty) {
        final distanceOfLast = sortedKNeighbors.last.distance;
        return sortedKNeighbors.map((neighbour) =>
            Neighbour(neighbour.distance / distanceOfLast, neighbour.label));
      }

      return sortedKNeighbors;
    });
  }

  int _findNewNeighbourIdx(
      double newNeighbourDist, List<Neighbour> sortedNeighbors) {
    var i = -1;

    for (final neighbour in sortedNeighbors) {
      if (newNeighbourDist < neighbour.distance) {
        return ++i;
      }
    }

    return i;
  }
}
