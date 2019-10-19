import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/kernel_function.dart';
import 'package:ml_algo/src/knn_solver/knn_solver.dart';
import 'package:ml_algo/src/knn_solver/neigbour.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class KnnClassifierImpl implements KnnClassifier {
  KnnClassifierImpl(
      this._trainingFeatures,
      this._trainingOutcomes,
      String targetName,
      this._kernelFn,
      this._k,
      this._distanceType,
      this._solverFn,
      this._dtype,
  ) : classNames = [targetName] {
    if (!_trainingFeatures.hasData) {
      throw Exception('Empty features matrix provided');
    }
    if (!_trainingOutcomes.hasData) {
      throw Exception('Empty outcomes matrix provided');
    }
    if (_trainingOutcomes.columnsNum > 1) {
      throw Exception('Invalid outcome matrix: it is expected to be a column '
          'vector, but a matrix of ${_trainingOutcomes.columnsNum} colums is '
          'given');
    }
    if (_trainingFeatures.rowsNum != _trainingOutcomes.rowsNum) {
      throw Exception('Number of feature records and number of associated '
          'outcomes must be equal');
    }
    if (_k <= 0 || _k > _trainingFeatures.rowsNum) {
      throw RangeError.value(_k, 'Parameter k should be within the range '
          '1..${_trainingFeatures.rowsNum} (both inclusive)');
    }
  }

  final Matrix _trainingFeatures;
  final Matrix _trainingOutcomes;
  final Distance _distanceType;
  final int _k;
  final FindKnnFn _solverFn;
  final KernelFn _kernelFn;
  final DType _dtype;

  @override
  final List<String> classNames;

  @override
  DataFrame predict(DataFrame features) {
    if (!features.toMatrix(_dtype).hasData) {
      throw Exception('No features provided');
    }

    if (features.toMatrix(_dtype).columnsNum != _trainingFeatures.columnsNum) {
      throw Exception('Invalid feature matrix: expected columns number: '
          '${_trainingFeatures.columnsNum}, given: '
          '${features.toMatrix(_dtype).columnsNum}');
    }

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

  Map<num, List<num>> _getLabelsToProbabilitiesMapping(DataFrame features) {
    final neighbours = _solverFn(
      _k,
      _trainingFeatures,
      _trainingOutcomes,
      features.toMatrix(_dtype),
      standardize: true,
      distance: _distanceType,
    );

    return neighbours.fold<Map<num, List<num>>>(
        {}, (allLabelsToProbabilities, kNeighbours) {
      final labelsToWeights = kNeighbours
          .fold<Map<num, num>>({}, _getLabelToWeightMapping);

      final sumOfAllWeights = labelsToWeights
          .values
          .reduce((sum, weight) => sum + weight);

      final labelsToProbabilities = labelsToWeights
          .map((key, weight) => MapEntry(key, weight / sumOfAllWeights));

      return allLabelsToProbabilities
        ..updateAll(
                (key, value) => (value ?? [])..add(labelsToProbabilities[key]));
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
    final weight = _kernelFn(neighbour.distance);
    return labelToWeightMapping
      ..update(
        neighbour.label.first,
            (totalWeight) => totalWeight + weight,
        ifAbsent: () => weight,
      );
  }
}
