import 'package:ml_algo/src/model_selection/split_indices_provider/split_indices_provider_factory_impl.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/split_indices_provider_type.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/k_fold_data_splitter.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/lpo_indices_provider.dart';
import 'package:test/test.dart';

void main() {
  group('SplitIndicesProviderFactoryImpl', () {
    const factory = SplitIndicesProviderFactoryImpl();

    test('should create k fold indices provider', () {
      expect(
        factory.createByType(SplitIndicesProviderType.kFold, numberOfFolds: 3),
        isA<KFoldIndicesProvider>(),
      );
    });

    test('should throw an exception if number of folds is not provided for '
        'k fold indices provider', () {
      expect(
        () => factory.createByType(SplitIndicesProviderType.kFold),
        throwsException,
      );
    });

    test('should create leave p out indices provider', () {
      expect(
        factory.createByType(SplitIndicesProviderType.lpo, p: 3),
        isA<LpoIndicesProvider>(),
      );
    });

    test('should throw an exception if `p` parameter is not provided for '
        'leave p out indices provider', () {
      expect(
            () => factory.createByType(SplitIndicesProviderType.lpo),
        throwsException,
      );
    });
  });
}
