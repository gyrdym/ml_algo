import 'package:ml_algo/src/algorithms/knn/kernel.dart';
import 'package:ml_algo/src/algorithms/knn/kernel_function_factory.dart';
import 'package:ml_algo/src/algorithms/knn/kernel_function_factory_impl.dart';
import 'package:ml_algo/src/algorithms/knn/kernel_type.dart';
import 'package:ml_algo/src/algorithms/knn/knn.dart';
import 'package:ml_algo/src/algorithms/knn/neigbour.dart';
import 'package:ml_algo/src/classifier/knn_classifier.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class KnnClassifierImpl implements KnnClassifier {
  KnnClassifierImpl(
      this._trainingFeatures,
      this._trainingOutcomes,
      String className, {
        int k,
        Distance distance = Distance.euclidean,
        FindKnnFn solverFn = findKNeighbours,
        Kernel kernel = Kernel.gaussian,
        DType dtype = DType.float32,

        KernelFunctionFactory kernelFnFactory =
          const KernelFunctionFactoryImpl(),
      }) :
        classNames = [className],
        _k = k,
        _distanceType = distance,
        _solverFn = solverFn,
        _dtype = dtype,
        _kernelFn = kernelFnFactory.createByType(kernel) {
    if (_trainingFeatures.rowsNum != _trainingOutcomes.rowsNum) {
      throw Exception('Number of features and number of outcomes must be'
          'equal');
    }
    if (_k > _trainingFeatures.rowsNum) {
      throw Exception('Parameter k should be less than or equal to the number '
          'of training observations');
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
