import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/model_selection/_injector.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/split_indices_provider.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/split_indices_provider_factory.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/split_indices_provider_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.dart';
import '../../mocks.mocks.dart';

void main() {
  group('CrossValidator', () {
    final data = DataFrame(
      [
        <num>[1, 2, 3, 4]
      ],
      headerExists: false,
      header: ['1', '2', '3', '4'],
    );

    late SplitIndicesProvider dataSplitter;
    late SplitIndicesProviderFactory dataSplitterFactory;

    setUp(() {
      dataSplitter = MockSplitIndicesProvider();
      dataSplitterFactory = createDataSplitterFactoryMock(dataSplitter);

      modelSelectionInjector
        ..clearAll()
        ..registerDependency<SplitIndicesProviderFactory>(
            () => dataSplitterFactory);
    });

    test(
        'should create k-fold cross validator and pass number of folds '
        'parameter into data splitter factory', () {
      CrossValidator.kFold(data, numberOfFolds: 10);

      verify(
        dataSplitterFactory.createByType(SplitIndicesProviderType.kFold,
            numberOfFolds: 10),
      ).called(1);
    });

    test(
        'should create k-fold cross validator and pass 5 as default value for '
        'number of folds parameter into data splitter factory', () {
      CrossValidator.kFold(data);

      verify(
        dataSplitterFactory.createByType(SplitIndicesProviderType.kFold,
            numberOfFolds: 5),
      ).called(1);
    });

    test(
        'should create leave-p-out cross validator and pass `p` parameter '
        'into data splitter factory', () {
      CrossValidator.lpo(data, 30);

      verify(
        dataSplitterFactory.createByType(SplitIndicesProviderType.lpo, p: 30),
      ).called(1);
    });
  });
}
