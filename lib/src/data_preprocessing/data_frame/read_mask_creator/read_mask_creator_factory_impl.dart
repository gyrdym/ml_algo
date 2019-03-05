import 'package:logging/logging.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/read_mask_creator/read_mask_creator.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/read_mask_creator/read_mask_creator_factory.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/read_mask_creator/read_mask_creator_impl.dart';

class DataFrameReadMaskCreatorFactoryImpl implements
    DataFrameReadMaskCreatorFactory {

  const DataFrameReadMaskCreatorFactoryImpl();

  @override
  DataFrameReadMaskCreator create(Logger logger) =>
      DataFrameReadMaskCreatorImpl(logger);
}
