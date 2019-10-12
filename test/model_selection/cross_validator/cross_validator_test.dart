import 'package:injector/injector.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/model_selection/data_splitter/data_splitter.dart';
import 'package:ml_algo/src/model_selection/data_splitter/data_splitter_factory.dart';
import 'package:ml_algo/src/model_selection/data_splitter/data_splitter_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.dart';

void main() {
  group('CrossValidator', () {
    final data = DataFrame(
      [<num>[1, 2, 3, 4]],
      headerExists: false,
      header: ['1', '2', '3', '4'],
    );

    DataSplitter dataSplitter;
    DataSplitterFactory dataSplitterFactory;

    setUp(() {
      dataSplitter = DataSplitterMock();
      dataSplitterFactory = createDataSplitterFactoryMock(dataSplitter);

      injector = Injector()
        ..registerDependency<DataSplitterFactory>((_) => dataSplitterFactory);
    });

    tearDown(() => injector = null);

    test('should create k-fold cross validator and pass number of folds '
        'parameter into data splitter factory', () {
      CrossValidator.kFold(data, ['4'], numberOfFolds: 10);

      verify(dataSplitterFactory
          .createByType(DataSplitterType.kFold, numberOfFolds: 10),
      ).called(1);
    });

    test('should create k-fold cross validator and pass 5 as default value for '
        'number of folds parameter into data splitter factory', () {
      CrossValidator.kFold(data, ['4']);

      verify(dataSplitterFactory
          .createByType(DataSplitterType.kFold, numberOfFolds: 5),
      ).called(1);
    });

    test('should create leave-p-out cross validator and pass `p` parameter '
        'into data splitter factory', () {
      CrossValidator.lpo(data, ['4'], 30);

      verify(dataSplitterFactory
          .createByType(DataSplitterType.lpo, p: 30),
      ).called(1);
    });
  });
}
