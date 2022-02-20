import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/classifier/_mixins/assessable_classifier_mixin.dart';
import 'package:ml_algo/src/classifier/knn_classifier/_injector.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_constants.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_factory.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_json_keys.dart';
import 'package:ml_algo/src/common/constants/common_json_keys.dart';
import 'package:ml_algo/src/common/json_converter/dtype_json_converter.dart';
import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';
import 'package:ml_algo/src/helpers/validate_class_label_list.dart';
import 'package:ml_algo/src/helpers/validate_test_features.dart';
import 'package:ml_algo/src/knn_kernel/kernel.dart';
import 'package:ml_algo/src/knn_kernel/kernel_json_converter.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type.dart';
import 'package:ml_algo/src/knn_solver/knn_solver.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_json_converter.dart';
import 'package:ml_algo/src/knn_solver/neigbour.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

part 'knn_classifier_impl.g.dart';

@JsonSerializable()
@KnnSolverJsonConverter()
@KernelJsonConverter()
@DTypeJsonConverter()
class KnnClassifierImpl
    with AssessableClassifierMixin, SerializableMixin
    implements KnnClassifier {
  KnnClassifierImpl(
    this.targetColumnName,
    this.classLabels,
    this.kernel,
    this.solver,
    this.classLabelPrefix,
    this.dtype, {
    this.schemaVersion = knnClassifierJsonSchemaVersion,
  }) {
    validateClassLabelList(classLabels);
  }

  factory KnnClassifierImpl.fromJson(Map<String, dynamic> json) =>
      _$KnnClassifierImplFromJson(json);

  @override
  Map<String, dynamic> toJson() => _$KnnClassifierImplToJson(this);

  @JsonKey(name: knnClassifierTargetColumnNameJsonKey)
  final String targetColumnName;

  @override
  @JsonKey(name: knnClassifierDTypeJsonKey)
  final DType dtype;

  @override
  Iterable<String> get targetNames => [targetColumnName];

  @JsonKey(name: knnClassifierClassLabelsJsonKey)
  final List<num> classLabels;

  @JsonKey(name: knnClassifierKernelJsonKey)
  final Kernel kernel;

  @JsonKey(name: knnClassifierSolverJsonKey)
  final KnnSolver solver;

  @JsonKey(name: knnClassifierClassLabelPrefixJsonKey)
  final String classLabelPrefix;

  @override
  num get positiveLabel => double.nan;

  @override
  num get negativeLabel => double.nan;

  @override
  int get k => solver.k;

  @override
  KernelType get kernelType => kernel.type;

  @override
  Distance get distanceType => solver.distanceType;

  @override
  @JsonKey(name: jsonSchemaVersionJsonKey)
  final int schemaVersion;

  @override
  DataFrame predict(DataFrame features) {
    validateTestFeatures(features, dtype);

    final labelsToProbabilities = _getLabelToProbabilityMapping(features);
    final labels = labelsToProbabilities.keys.toList();
    final predictedOutcomes =
        _getProbabilityMatrix(labelsToProbabilities).rows.map((probabilities) {
      // TODO: extract max element index search logic to ml_linalg
      // TODO: fix corner cases with NaN and Infinity
      final maxProbability = probabilities.max();
      final maxProbabilityIndex =
          probabilities.toList().indexOf(maxProbability);

      if (maxProbabilityIndex == -1) {
        print('KnnClassifier error: cannot find max probability, '
            'max probability is $maxProbability');

        return labels.first;
      }

      return labels[maxProbabilityIndex];
    }).toList();

    final outcomesAsVector = Vector.fromList(predictedOutcomes, dtype: dtype);

    return DataFrame.fromMatrix(
      Matrix.fromColumns([outcomesAsVector], dtype: dtype),
      header: targetNames,
    );
  }

  @override
  DataFrame predictProbabilities(DataFrame features) {
    final labelsToProbabilities = _getLabelToProbabilityMapping(features);
    final probabilityMatrix = _getProbabilityMatrix(labelsToProbabilities);
    final header = labelsToProbabilities.keys.map((label) => [
          classLabelPrefix.trim(),
          label.toString().trim()
        ].where((element) => element.isNotEmpty).join(' '));

    return DataFrame.fromMatrix(probabilityMatrix, header: header);
  }

  @override
  KnnClassifier retrain(DataFrame data) {
    return knnClassifierInjector.get<KnnClassifierFactory>().create(
          data,
          targetColumnName,
          k,
          kernelType,
          distanceType,
          classLabelPrefix,
          dtype,
        );
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
  /// feature record from the test feature matrix
  Map<num, List<num>> _getLabelToProbabilityMapping(DataFrame features) {
    final kNeighbourGroups = solver.findKNeighbours(features.toMatrix(dtype));
    final classLabelsAsSet = Set<num>.from(classLabels);

    return kNeighbourGroups.fold<Map<num, List<num>>>({},
        (allLabelsToProbabilities, kNeighbours) {
      final labelsToWeights =
          kNeighbours.fold<Map<num, num>>({}, (mapping, neighbour) {
        if (!classLabelsAsSet.contains(neighbour.label.first)) {
          throw Exception('Wrong KNN solver provided: unexpected neighbour '
              'class label - ${neighbour.label.first}');
        }
        return _updateLabelToWeightMapping(mapping, neighbour);
      });

      final sumOfAllWeights =
          labelsToWeights.values.reduce((sum, weight) => sum + weight);

      final labelsToProbabilities = labelsToWeights
          .map((key, weight) => MapEntry(key, weight / sumOfAllWeights));

      final areLabelsEquiprobable =
          _areLabelsEquiprobable(labelsToProbabilities.values);

      // if labels are equiprobable, make the first neighbour's label
      // probability equal to 1 and probabilities of the rest neighbour labels -
      // equal to 0
      classLabels.forEach((label) {
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
    final probabilityVectors = allLabelsToProbabilities.values
        .map((probabilities) => Vector.fromList(probabilities, dtype: dtype))
        .toList(growable: false);

    return Matrix.fromColumns(probabilityVectors, dtype: dtype);
  }

  Map<num, num> _updateLabelToWeightMapping(
    Map<num, num> labelToWeightMapping,
    Neighbour<Vector> neighbour,
  ) {
    final weight = kernel.getWeightByDistance(neighbour.distance);
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
