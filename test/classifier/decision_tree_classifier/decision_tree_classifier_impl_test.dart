import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/classifier/classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/_injector.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_constants.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_factory.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_impl.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_json_keys.dart';
import 'package:ml_algo/src/common/constants/common_json_keys.dart';
import 'package:ml_algo/src/common/exception/outdated_json_schema_exception.dart';
import 'package:ml_algo/src/di/common/init_common_module.dart';
import 'package:ml_algo/src/di/dependency_keys.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/metric/metric_factory.dart';
import 'package:ml_algo/src/model_selection/model_assessor/model_assessor.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node_json_keys.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/dtype_to_json.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../helpers.dart';
import '../../mocks.dart';
import '../../mocks.mocks.dart';

void main() {
  group('DecisionTreeClassifierImpl', () {
    final minError = 0.2;
    final minSamplesCount = 3;
    final maxDepth = 5;
    final classifierAssessorMock = MockClassifierAssessor();
    final sample1 = Vector.fromList([1, 2, 3]);
    final sample2 = Vector.fromList([10, 20, 30]);
    final sample3 = Vector.fromList([100, 200, 300]);
    final label1 = 100;
    final label2 = 300;
    final label3 = 200;
    final sample1WithLabel = Vector.fromList([...sample1, label1]);
    final sample2WithLabel = Vector.fromList([...sample2, label2]);
    final sample3WithLabel = Vector.fromList([...sample3, label3]);
    final predictedBinarizedLabels = [
      [0, 0, 1],
      [1, 0, 0],
      [0, 1, 0],
    ];
    final originalBinarizedLabels = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    final learnedLeafLabel1 = TreeLeafLabel(label3, probability: 0.7);
    final learnedLeafLabel2 = TreeLeafLabel(label1, probability: 0.55);
    final learnedLeafLabel3 = TreeLeafLabel(label2, probability: 0.5);
    final features = Matrix.fromRows([
      sample1,
      sample2,
      sample3,
    ]);
    final labelledFeatures = Matrix.fromRows([
      sample1WithLabel,
      sample2WithLabel,
      sample3WithLabel,
    ]);
    final unlabelledFeaturesFrame = DataFrame.fromMatrix(features);
    final labelledFeaturesFrame = DataFrame.fromMatrix(labelledFeatures);
    final targetColumnName = labelledFeaturesFrame.header.last;
    final predictedLabelsFrame = DataFrame(
        [
          <dynamic>[label3],
          <dynamic>[label1],
          <dynamic>[label2],
        ],
        headerExists: false,
        header: [targetColumnName]);
    final predictedBinarizedLabelsFrame = DataFrame(predictedBinarizedLabels,
        headerExists: false, header: [targetColumnName]);
    final originalBinarizedLabelsFrame = DataFrame(originalBinarizedLabels,
        headerExists: false, header: [targetColumnName]);
    final rootNodeJson = {
      childrenJsonKey: <Map<String, dynamic>>[],
      levelJsonKey: 1,
    };
    final classifier32Json = {
      decisionTreeClassifierMinErrorJsonKey: minError,
      decisionTreeClassifierMaxDepthJsonKey: maxDepth,
      decisionTreeClassifierMinSamplesCountJsonKey: minSamplesCount,
      decisionTreeClassifierDTypeJsonKey: dTypeToJson(DType.float32),
      decisionTreeClassifierTargetColumnNameJsonKey: targetColumnName,
      decisionTreeClassifierTreeRootNodeJsonKey: rootNodeJson,
      jsonSchemaVersionJsonKey: decisionTreeClassifierJsonSchemaVersion,
    };
    final classifier64Json = {
      decisionTreeClassifierMinErrorJsonKey: minError,
      decisionTreeClassifierMaxDepthJsonKey: maxDepth,
      decisionTreeClassifierMinSamplesCountJsonKey: minSamplesCount,
      decisionTreeClassifierDTypeJsonKey: dTypeToJson(DType.float64),
      decisionTreeClassifierTargetColumnNameJsonKey: targetColumnName,
      decisionTreeClassifierTreeRootNodeJsonKey: rootNodeJson,
      jsonSchemaVersionJsonKey: decisionTreeClassifierJsonSchemaVersion,
    };
    final treeRootMock = createRootNodeMock({
      sample1: learnedLeafLabel1,
      sample2: learnedLeafLabel2,
      sample3: learnedLeafLabel3,
    }, rootNodeJson);
    final metricFactoryMock = MockMetricFactory();
    final metricMock = MockMetric();
    final encoderFactoryMock = MockEncoderFactory();
    final encoderMock = MockEncoder();
    final encodedLabelsFrames = [
      predictedBinarizedLabelsFrame,
      originalBinarizedLabelsFrame,
    ];
    final retrainingDataFrame = DataFrame([
      [1, 2, 3, 4],
    ]);
    final classifierFactoryMock = MockDecisionTreeClassifierFactory();
    final retrainedClassifier = MockDecisionTreeClassifier();
    var encoderCallIteration = 0;

    late DecisionTreeClassifierImpl classifier32;
    late DecisionTreeClassifierImpl classifier64;

    setUp(() {
      when(
        metricFactoryMock.createByType(
          argThat(
            isA<MetricType>(),
          ),
        ),
      ).thenReturn(metricMock);

      when(
        encoderFactoryMock.create(
          any,
          any,
        ),
      ).thenReturn(encoderMock);

      when(
        encoderMock.process(
          any,
        ),
      ).thenAnswer(
            (_) => encodedLabelsFrames[encoderCallIteration++],
      );

      when(
        classifierFactoryMock.create(
          any,
          any,
          any,
          any,
          any,
          any,
        ),
      ).thenReturn(retrainedClassifier);

      when(
        classifierAssessorMock.assess(
          any,
          any,
          any,
        ),
      ).thenReturn(1.0);

      injector
        ..registerDependency<ModelAssessor<Classifier>>(
            () => classifierAssessorMock)
        ..registerDependency<EncoderFactory>(() => encoderFactoryMock.create,
            dependencyName: oneHotEncoderFactoryKey)
        ..registerSingleton<MetricFactory>(() => metricFactoryMock);
      decisionTreeInjector
        .registerSingleton<DecisionTreeClassifierFactory>(
                () => classifierFactoryMock);

      classifier32 = DecisionTreeClassifierImpl(
        minError,
        minSamplesCount,
        maxDepth,
        treeRootMock,
        targetColumnName,
        DType.float32,
      );

      classifier64 = DecisionTreeClassifierImpl(
        minError,
        minSamplesCount,
        maxDepth,
        treeRootMock,
        targetColumnName,
        DType.float64,
      );
    });

    tearDown(() {
      reset(metricFactoryMock);
      reset(metricMock);
      reset(encoderFactoryMock);
      reset(encoderMock);
      encoderCallIteration = 0;

      injector.clearAll();
      decisionTreeInjector.clearAll();
    });

    test('should persist hyperparameters for float32-based classifier', () {
      expect(classifier32.maxDepth, maxDepth);
      expect(classifier32.minSamplesCount, minSamplesCount);
      expect(classifier32.minError, minError);
    });

    test('should persist hyperparameters for float64-based classifier', () {
      expect(classifier64.maxDepth, maxDepth);
      expect(classifier64.minSamplesCount, minSamplesCount);
      expect(classifier64.minError, minError);
    });

    test('should predict labels for passed unlabelled features dataframe', () {
      final actual = classifier32.predict(unlabelledFeaturesFrame);

      expect(actual.toMatrix(), predictedLabelsFrame.toMatrix());
    });

    test('should return predicted labels with a proper header', () {
      final actual = classifier32.predict(unlabelledFeaturesFrame);

      expect(actual.header, classifier32.targetNames);
    });

    test('should return data frame with empty header if input matrix is '
        'empty', () {
      final predictedClasses = classifier32.predict(DataFrame([<num>[]]));

      expect(predictedClasses.header, isEmpty);
    });

    test('should return data frame with empty matrix if input feature matrix '
        'is empty', () {
      final predictedClasses = classifier32.predict(DataFrame([<num>[]]));

      expect(predictedClasses.toMatrix(), isEmpty);
    });

    test('should return data frame with probabilities for each class '
        'label', () {
      final predictedLabels =
          classifier32.predictProbabilities(unlabelledFeaturesFrame);

      expect(
        predictedLabels.toMatrix(),
        iterable2dAlmostEqualTo([
          [learnedLeafLabel1.probability.toDouble()],
          [learnedLeafLabel2.probability.toDouble()],
          [learnedLeafLabel3.probability.toDouble()],
        ]),
      );
    });

    test('should serialize (dtype is float32)', () {
      final data = classifier32.toJson();

      expect(data, equals(classifier32Json));
      verify(treeRootMock.toJson()).called(1);
    });

    test('should serialize (dtype is float64)', () {
      final data = classifier64.toJson();

      expect(data, equals(classifier64Json));
      verify(treeRootMock.toJson()).called(1);
    });

    test('should restore dtype field from json (dtype is float32)', () {
      final classifier = DecisionTreeClassifierImpl.fromJson(classifier32Json);

      expect(classifier.dtype, equals(DType.float32));
    });

    test('should restore dtype field from json (dtype is float64)', () {
      final classifier = DecisionTreeClassifierImpl.fromJson(classifier64Json);

      expect(classifier.dtype, equals(DType.float64));
    });

    test('should be restored from json', () {
      final classifier = DecisionTreeClassifierImpl.fromJson(classifier32Json);

      expect(classifier.targetColumnName, equals(targetColumnName));
      expect(classifier.treeRootNode, isNotNull);
    });

    test('should call classifier assessor, dtype=DType.float32', () {
      final metricType = MetricType.precision;

      classifier32.assess(labelledFeaturesFrame, metricType);

      verify(
        classifierAssessorMock.assess(
          classifier32,
          metricType,
          labelledFeaturesFrame,
        ),
      ).called(1);
    });

    test('should call classifier factory while retraining the model', () {
      classifier32.retrain(retrainingDataFrame);

      verify(classifierFactoryMock.create(
        retrainingDataFrame,
        targetColumnName,
        DType.float32,
        minError,
        minSamplesCount,
        maxDepth,
      )).called(1);
    });

    test('should return a new instance for the retrained model', () {
      final retrainedModel = classifier32.retrain(retrainingDataFrame);

      expect(retrainedModel, same(retrainedClassifier));
      expect(retrainedModel, isNot(same(classifier32)));
    });

    test('should throw exception if the model schema is outdated, '
        'schemaVersion is null', () {
      final model = DecisionTreeClassifierImpl(
        minError,
        minSamplesCount,
        maxDepth,
        treeRootMock,
        targetColumnName,
        DType.float64,
        schemaVersion: null,
      );

      expect(() => model.retrain(retrainingDataFrame),
          throwsA(isA<OutdatedJsonSchemaException>()));
    });
  });
}

TreeNode createRootNodeMock(Map<Vector, TreeLeafLabel> samplesByLabel,
    [Map<String, dynamic> jsonMock = const <String, dynamic>{}]) {
  final rootMock = MockTreeNode();
  final children = <TreeNode>[];

  when(rootMock.isLeaf).thenReturn(false);

  samplesByLabel.forEach((sample, leafLabel) {
    final node = MockTreeNode();

    when(node.label).thenReturn(leafLabel);
    when(node.isLeaf).thenReturn(true);

    samplesByLabel.forEach((otherSample, _) =>
        when(node.isSamplePassed(otherSample))
            .thenReturn(sample == otherSample));

    children.add(node);
  });

  when(rootMock.children).thenReturn(children);
  when(rootMock.toJson()).thenReturn(jsonMock);

  return rootMock;
}
