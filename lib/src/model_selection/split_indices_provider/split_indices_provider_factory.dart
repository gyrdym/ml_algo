import 'package:ml_algo/src/model_selection/split_indices_provider/split_indices_provider_type.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/split_indices_provider.dart';

abstract class SplitIndicesProviderFactory {
  SplitIndicesProvider createByType(SplitIndicesProviderType splitterType, {
    int numberOfFolds,
    int p,
  });
}
