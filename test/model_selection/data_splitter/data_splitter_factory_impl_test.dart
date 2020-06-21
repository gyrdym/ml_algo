import 'package:ml_algo/src/model_selection/split_indices_provider/data_splitter_factory_impl.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/data_splitter_type.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/k_fold_data_splitter.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/leave_p_out_data_splitter.dart';
import 'package:test/test.dart';

void main() {
  group('DataSplitterFactoryImpl', () {
    const factory = DataSplitterFactoryImpl();

    test('should create k fold data splitter', () {
      expect(
        factory.createByType(DataSplitterType.kFold, numberOfFolds: 3),
        isA<KFoldDataSplitter>(),
      );
    });

    test('should throw an exception if number of folds is not provided for '
        'k fold data splitter', () {
      expect(
        () => factory.createByType(DataSplitterType.kFold),
        throwsException,
      );
    });

    test('should create leave p out data splitter', () {
      expect(
        factory.createByType(DataSplitterType.lpo, p: 3),
        isA<LeavePOutDataSplitter>(),
      );
    });

    test('should throw an exception if `p` parameter is not provided for '
        'leave p out data splitter', () {
      expect(
            () => factory.createByType(DataSplitterType.lpo),
        throwsException,
      );
    });
  });
}
