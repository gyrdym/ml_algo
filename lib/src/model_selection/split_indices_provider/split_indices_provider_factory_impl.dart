import 'package:ml_algo/src/model_selection/split_indices_provider/split_indices_provider_factory.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/split_indices_provider_type.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/k_fold_data_splitter.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/lpo_indices_provider.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/split_indices_provider.dart';

class SplitIndicesProviderFactoryImpl implements SplitIndicesProviderFactory {
  const SplitIndicesProviderFactoryImpl();

  @override
  SplitIndicesProvider createByType(
    SplitIndicesProviderType splitterType, {
    int? numberOfFolds,
    int? p,
  }) {
    switch (splitterType) {
      case SplitIndicesProviderType.kFold:
        if (numberOfFolds == null) {
          throw Exception('Number of folds is not defined for K-fold splitter');
        }
        return KFoldIndicesProvider(numberOfFolds);

      case SplitIndicesProviderType.lpo:
        if (p == null) {
          throw Exception('`p` parameter is not defined for leave-p-out '
              'splitter');
        }
        return LpoIndicesProvider(p);

      default:
        throw UnimplementedError('Splitter of type $splitterType is not '
            'implemented yet');
    }
  }
}
