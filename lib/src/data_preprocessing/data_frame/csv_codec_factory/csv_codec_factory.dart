import 'package:csv/csv.dart';

abstract class CsvCodecFactory {
  CsvCodec create({
    String fieldDelimiter,
    String eol,
  });
}
