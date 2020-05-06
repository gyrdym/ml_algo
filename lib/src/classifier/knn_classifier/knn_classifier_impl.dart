import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier.dart';
import 'package:ml_algo/src/helpers/validate_class_label_list.dart';
import 'package:ml_algo/src/helpers/validate_test_features.dart';
import 'package:ml_algo/src/knn_kernel/kernel.dart';
import 'package:ml_algo/src/knn_solver/knn_solver.dart';
import 'package:ml_algo/src/knn_solver/neigbour.dart';
import 'package:ml_algo/src/predictor/assessable_predictor_mixin.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class KnnClassifierImpl with AssessablePredictorMixin implements KnnClassifier {
  KnnClassifierImpl(
      this._targetColumnName,
      this._classLabels,
      this._kernel,
      this._solver,
      this.dtype,
  ) {
    validateClassLabelList(_classLabels);
  }

  final String _targetColumnName;

  @override
  final DType dtype;

  final List<num> _classLabels;
  final Kernel _kernel;
  final KnnSolver _solver;
  final String _columnPrefix = 'Class label';

  @override
  DataFrame predict(DataFrame features) {
    validateTestFeatures(features, dtype);

    final labelsToProbabilities = _getLabelToProbabilityMapping(features);
    final labels = labelsToProbabilities.keys.toList();
    final predictedOutcomes = _getProbabilityMatrix(labelsToProbabilities)
        .rows
        .map((row) => labels[row.toList().indexOf(row.max())])
        .toList();

    final outcomesAsVector = Vector.fromList(predictedOutcomes, dtype: dtype);

    return DataFrame.fromMatrix(
      Matrix.fromColumns([outcomesAsVector], dtype: dtype),
      header: [_targetColumnName],
    );
  }

  @override
  DataFrame predictProbabilities(DataFrame features) {
    final labelsToProbabilities = _getLabelToProbabilityMapping(features);
    final probabilityMatrix = _getProbabilityMatrix(labelsToProbabilities);

    final header = labelsToProbabilities
        .keys
        .map((label) => '${_columnPrefix} ${label.toString()}');

    return DataFrame.fromMatrix(probabilityMatrix, header: header);
  }

  /// Returns a map of the following format:
  ///
  /// ```
  /// class_1_label: probability_1, probability_2, ..., probability_n
  /// class_2_label: probability_1, probability_2, ..., probability_n
  /// ...
  /// class_n_label: probability_1, probability_2, ..., probability_n
  /// ```
  ///
  /// This may be interpreted as a table of probabilities:
  ///
  /// ```
  /// class_1_label   class_2_label  ...  class_n_label
  /// -------------------------------------------------
  /// probability_1   probability_1       probability_1
  /// probability_2   probability_2       probability_2
  ///      ...             ...                 ...
  /// probability_n   probability_n       probability_n
  /// ```
  ///
  /// where each row is a classes probability distribution for the appropriate
  /// feature record from test feature matrix
  Map<num, List<num>> _getLabelToProbabilityMapping(DataFrame features) {
    final kNeighbourGroups = _solver.findKNeighbours(features.toMatrix(dtype));
    final classLabelsAsSet = Set<num>.from(_classLabels);

    return kNeighbourGroups.fold<Map<num, List<num>>>(
        {}, (allLabelsToProbabilities, kNeighbours) {

      final labelsToWeights = kNeighbours.fold<Map<num, num>>(
          {}, (mapping, neighbour) {
        if (!classLabelsAsSet.contains(neighbour.label.first)) {
          throw Exception('Wrong KNN solver provided: unexpected neighbour '
              'class label - ${neighbour.label.first}');
        }
        return _updateLabelToWeightMapping(mapping, neighbour);
      });

      final sumOfAllWeights = labelsToWeights
          .values
          .reduce((sum, weight) => sum + weight);

      final labelsToProbabilities = labelsToWeights
          .map((key, weight) => MapEntry(key, weight / sumOfAllWeights));

      final areLabelsEquiprobable = _areLabelsEquiprobable(
          labelsToProbabilities.values);

      // if labels are equiprobable, make the first neighbour's label
      // probability equal to 1 and probabilities of the rest neighbour labels -
      // equal to 0
      _classLabels.forEach((label) {
        final probability = areLabelsEquiprobable
            ? label == kNeighbours.first.label.first
                ? 1
                : 0
            : labelsToProbabilities[label] ?? 0;

        allLabelsToProbabilities.update(
          label,
          (probabilities) => probabilities..add(probability),
          ifAbsent: () => [probability],
        );
      });

      return allLabelsToProbabilities;
    });
  }

  Matrix _getProbabilityMatrix(Map<num, List<num>> allLabelsToProbabilities) {
    final probabilityVectors = allLabelsToProbabilities
        .values
        .map((probabilities) => Vector.fromList(probabilities, dtype: dtype))
        .toList(growable: false);

    return Matrix
        .fromColumns(probabilityVectors, dtype: dtype);
  }

  Map<num, num> _updateLabelToWeightMapping(
      Map<num, num> labelToWeightMapping,
      Neighbour<Vector> neighbour,
  ) {
    final weight = _kernel.getWeightByDistance(neighbour.distance);
    return labelToWeightMapping
      ..update(
        neighbour.label.first,
        (totalWeight) => totalWeight + weight,
        ifAbsent: () => weight,
      );
  }

  bool _areLabelsEquiprobable(Iterable<num> labelProbabilities) =>
      Set<num>.from(labelProbabilities).length == 1;
}
