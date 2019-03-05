import 'package:logging/logging.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/read_mask_creator/read_mask_creator.dart';

abstract class DataFrameReadMaskCreatorFactory {
  DataFrameReadMaskCreator create(Logger logger);
}
