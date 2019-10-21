import 'package:ml_algo/src/_mixin/data_validation_mixin.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier.dart';
import 'package:ml_algo/src/knn_kernel/kernel.dart';
import 'package:ml_algo/src/knn_solver/knn_solver.dart';
import 'package:ml_algo/src/knn_solver/neigbour.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class KnnClassifierImpl with DataValidationMixin implements KnnClassifier {
  KnnClassifierImpl(
      String targetName,
      this._classLabels,
      this._kernel,
      this._solver,
      this._dtype,
  ) : classNames = [targetName] {
    if (_classLabels.isEmpty) {
      throw Exception('Empty class label list provided');
    }
  }

  final KnnSolver _solver;
  final Kernel _kernel;
  final DType _dtype;

  @override
  final List<String> classNames;

  final List<num> _classLabels;

  @override
  DataFrame predict(DataFrame features) {
    validateTestFeatures(features, _dtype);

    final labelsToProbabilities = _getLabelsToProbabilitiesMapping(features);
    final labels = labelsToProbabilities.keys.toList();
    final predictedOutcomes = _getProbabilityMatrix(labelsToProbabilities)
        .rows
        .map((row) => labels[row.toList().indexOf(row.max())])
        .toList();

    final outcomesAsVector = Vector.fromList(predictedOutcomes, dtype: _dtype);

    return DataFrame.fromMatrix(
      Matrix.fromColumns([outcomesAsVector], dtype: _dtype),
      header: classNames,
    );
  }

  @override
  DataFrame predictProbabilities(DataFrame features) {
    final labelsToProbabilities = _getLabelsToProbabilitiesMapping(features);
    final probabilityMatrix = _getProbabilityMatrix(labelsToProbabilities);

    final header = labelsToProbabilities
        .keys
        .map((label) => label.toString());

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
  Map<num, List<num>> _getLabelsToProbabilitiesMapping(DataFrame features) {
    final neighbours = _solver.findKNeighbours(features.toMatrix(_dtype));

    return neighbours.fold<Map<num, List<num>>>(
        {}, (allLabelsToProbabilities, kNeighbours) {
      final labelsToWeights = kNeighbours
          .fold<Map<num, num>>({}, _getLabelToWeightMapping);

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
        .map((probabilities) => Vector.fromList(probabilities, dtype: _dtype))
        .toList(growable: false);

    return Matrix
        .fromColumns(probabilityVectors, dtype: _dtype);
  }

  Map<num, num> _getLabelToWeightMapping(
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
