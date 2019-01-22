import 'package:logging/logging.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/read_mask_creator/read_mask_creator.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/read_mask_creator/read_mask_creator_factory.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/read_mask_creator/read_mask_creator_impl.dart';

class MLDataReadMaskCreatorFactoryImpl implements MLDataReadMaskCreatorFactory {
  const MLDataReadMaskCreatorFactoryImpl();

  @override
  MLDataReadMaskCreator create(Logger logger) => MLDataReadMaskCreatorImpl(logger);
}
