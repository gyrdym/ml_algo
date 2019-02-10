import 'package:logging/logging.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/read_mask_creator/read_mask_creator.dart';

abstract class MLDataReadMaskCreatorFactory {
  MLDataReadMaskCreator create(Logger logger);
}
