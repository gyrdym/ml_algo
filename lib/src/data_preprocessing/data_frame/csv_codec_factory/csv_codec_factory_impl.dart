import 'package:csv/csv.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/csv_codec_factory/csv_codec_factory.dart';

class CsvCodecFactoryImpl implements CsvCodecFactory {
  const CsvCodecFactoryImpl();

  @override
  CsvCodec create({String fieldDelimiter, String eol}) =>
      CsvCodec(fieldDelimiter: fieldDelimiter, eol: eol);
}
