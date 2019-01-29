import 'package:ml_algo/src/classifier/labels_processor/labels_processor.dart';
import 'package:ml_algo/src/classifier/labels_processor/labels_processor_factory.dart';
import 'package:ml_algo/src/classifier/labels_processor/labels_processor_impl.dart';

class LabelsProcessorFactoryImpl implements LabelsProcessorFactory {
  const LabelsProcessorFactoryImpl();

  @override
  LabelsProcessor create() => LabelsProcessorImpl();
}
